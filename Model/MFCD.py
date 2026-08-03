import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from EduCDM import CDM


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class NetWithFluency(nn.Module):
    """
    Unified Fluency Proxy version.

    Main task:
      - predict correctness p(correct)

    Fluency inputs:
      - opportunity
      - acc_prior
      - stability
    """

    def __init__(
        self,
        knowledge_n,
        exer_n,
        student_n,
        prednet_len1=512,
        prednet_len2=256,
        dropout=0.5,
        rt_cap_ms=120000.0,
        flu_latent_dim=32,
        gamma_init=1.0,
        fluency_lambda=0.3,
    ):
        super(NetWithFluency, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.rt_cap_ms = float(rt_cap_ms)

        self.student_emb = nn.Embedding(self.emb_num, self.knowledge_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)

        self.prednet_full1 = PosLinear(self.knowledge_dim, prednet_len1)
        self.drop_1 = nn.Dropout(p=dropout)
        self.prednet_full2 = PosLinear(prednet_len1, prednet_len2)
        self.drop_2 = nn.Dropout(p=dropout)
        self.prednet_full3 = PosLinear(prednet_len2, 1)

        self.beh_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, flu_latent_dim),
            nn.ReLU(),
        )

        self.student_flu_bias = nn.Embedding(self.emb_num, flu_latent_dim)

        self.flu_score_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.opp_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.acc_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.sta_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.fuse_gate = nn.Sequential(
            nn.Linear(flu_latent_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.fluency_lambda = float(fluency_lambda)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    @staticmethod
    def _safe_log1p(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x)

    @staticmethod
    def _to_col(t: torch.Tensor) -> torch.Tensor:
        return t.view(-1, 1) if t.dim() == 1 else t

    def forward(
        self,
        stu_id,
        input_exercise,
        input_knowledge_point,
        opportunity,
        acc_prior,
        stability,
        return_aux: bool = False,
    ):
        stu_id = torch.clamp(stu_id, 0, self.emb_num - 1)
        input_exercise = torch.clamp(input_exercise, 0, self.exer_n - 1)

        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))

        x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point.float()
        x = self.drop_1(torch.sigmoid(self.prednet_full1(x)))
        x = self.drop_2(torch.sigmoid(self.prednet_full2(x)))
        o_logit = self.prednet_full3(x)
        o = torch.sigmoid(o_logit)

        opp = self._to_col(opportunity.float())
        acc = self._to_col(acc_prior.float())
        sta = self._to_col(stability.float())

        f_opp = self._safe_log1p(opp)
        f_acc = torch.clamp(acc, 0.0, 1.0)
        f_sta = 1.0 - torch.clamp(4.0 * torch.clamp(sta, min=0.0), 0.0, 1.0)

        beh = torch.cat([f_opp, f_acc, f_sta], dim=1)

        z = self.beh_encoder(beh)
        z = z + torch.tanh(self.student_flu_bias(stu_id))
        flu = torch.sigmoid(self.flu_score_head(z))

        opp_tgt = f_opp
        acc_tgt = f_acc
        sta_tgt = f_sta

        opp_hat = self.opp_head(z)
        acc_hat = torch.sigmoid(self.acc_head(z))
        sta_hat = torch.sigmoid(self.sta_head(z))

        gate_in = torch.cat(
            [z, e_difficulty, k_difficulty.mean(dim=1, keepdim=True)],
            dim=1
        )
        gate = torch.sigmoid(self.fuse_gate(gate_in))

        gamma = F.softplus(self.gamma)
        delta = (flu - 0.5) * 4.0
        p_logit = o_logit + gamma * gate * delta
        p = torch.sigmoid(p_logit)

        if not return_aux:
            return p.view(-1)

        return {
            "p": p.view(-1),
            "o": o.view(-1),
            "flu": flu.view(-1),
            "gate": gate.view(-1),
            "opp_hat": opp_hat.view(-1),
            "acc_hat": acc_hat.view(-1),
            "sta_hat": sta_hat.view(-1),
            "opp_tgt": opp_tgt.view(-1),
            "acc_tgt": acc_tgt.view(-1),
            "sta_tgt": sta_tgt.view(-1),
        }


class MFCD(CDM):
    def __init__(
        self,
        knowledge_n,
        exer_n,
        student_n,
        lr=0.002,
        fluency_lambda=0.3,
        dropout=0.5,
        lambda_rt=0.1,
        lambda_hint=0.05,
        lambda_att=0.05,
        rt_cap_ms=120000.0,
        flu_latent_dim=32,
    ):
        super(MFCD, self).__init__()
        self.lr = lr
        self.fluency_lambda = float(fluency_lambda)

        self.lambda_rt = float(lambda_rt)
        self.lambda_hint = float(lambda_hint)
        self.lambda_att = float(lambda_att)

        self.net = NetWithFluency(
            knowledge_n,
            exer_n,
            student_n,
            dropout=dropout,
            fluency_lambda=fluency_lambda,
            rt_cap_ms=rt_cap_ms,
            flu_latent_dim=flu_latent_dim,
        )

        self.loss_y = nn.BCELoss()
        self.loss_opp = nn.MSELoss()
        self.loss_acc = nn.SmoothL1Loss()
        self.loss_sta = nn.SmoothL1Loss()

    def train(self, train_data, test_data=None, epoch=10, device="cuda", lr=None, silence=False):
        self.net = self.net.to(device)
        self.net.train()
        optimizer = optim.Adam(self.net.parameters(), lr=(lr or self.lr))

        for epoch_i in range(epoch):
            losses = []
            ly_list, lopp_list, lacc_list, lsta_list = [], [], [], []

            for batch_data in tqdm(train_data, f"Epoch {epoch_i}", disable=silence):
                user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch_data

                user_id = user_id.to(device)
                item_id = item_id.to(device)
                knowledge_emb = knowledge_emb.to(device)
                y = y.to(device).float()

                opp = opp.to(device)
                acc_prior = acc_prior.to(device)
                stability = stability.to(device)

                out = self.net(
                    user_id, item_id, knowledge_emb,
                    opp, acc_prior, stability,
                    return_aux=True
                )

                ly = self.loss_y(out["p"], y)
                lopp = self.loss_opp(out["opp_hat"], out["opp_tgt"])
                lacc = self.loss_acc(out["acc_hat"], out["acc_tgt"])
                lsta = self.loss_sta(out["sta_hat"], out["sta_tgt"])

                loss = ly + self.lambda_rt * lopp + self.lambda_hint * lacc + self.lambda_att * lsta

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                ly_list.append(ly.item())
                lopp_list.append(lopp.item())
                lacc_list.append(lacc.item())
                lsta_list.append(lsta.item())

            print(
                f"[Epoch {epoch_i}] loss={float(np.mean(losses)):.6f} "
                f"(y={float(np.mean(ly_list)):.6f}, "
                f"opp={float(np.mean(lopp_list)):.6f}, "
                f"acc={float(np.mean(lacc_list)):.6f}, "
                f"sta={float(np.mean(lsta_list)):.6f})"
            )

            if test_data is not None:
                auc, acc = self.eval(test_data, device=device, silence=silence)
                print(f"[Epoch {epoch_i}] auc: {auc:.6f}, accuracy: {acc:.6f}")

    @torch.no_grad()
    def eval(self, test_data, device="cpu", silence=False):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []

        for batch_data in tqdm(test_data, "Evaluating", disable=silence):
            user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch_data

            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            opp = opp.to(device)
            acc_prior = acc_prior.to(device)
            stability = stability.to(device)

            pred = self.net(
                user_id, item_id, knowledge_emb,
                opp, acc_prior, stability,
                return_aux=False
            )
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath, map_location=None):
        self.net.load_state_dict(torch.load(filepath, map_location=map_location))
        logging.info("load parameters from %s" % filepath)
