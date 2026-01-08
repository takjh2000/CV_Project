from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =========================
# Model Class (학습때와 동일해야 함)
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=33, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x, lengths):
        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        if self.lstm.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B,2H)
        else:
            h = h_n[-1]  # (B,H)

        return self.head(h)

# =========================
# 외부 호출용 함수 1: 모델 로드
# =========================
def load_model(ckpt_path, device):
    """
    체크포인트 경로를 받아 모델을 메모리에 로드하고 반환합니다.
    """
    # Path 객체로 변환
    ckpt_path = Path(ckpt_path)
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 모델 설정 로드
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    # 모델 초기화
    model = LSTMClassifier(
        input_size=int(cfg.get("input_size", 33)),
        hidden_size=int(cfg.get("hidden", 128)),
        num_layers=int(cfg.get("num_layers", 2)),
        bidirectional=bool(cfg.get("bidir", True)),
        dropout=float(cfg.get("dropout", 0.2)),
    ).to(device)
    
    # 가중치 로드
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    return model, cfg

# =========================
# 외부 호출용 함수 2: 단일 파일 추론
# =========================
def predict_single(model, csv_path, device, cfg):
    """
    로드된 모델과 CSV 경로를 받아 추론 결과를 딕셔너리로 반환합니다.
    """
    # CSV 로드
    df = pd.read_csv(csv_path)

    # frame 컬럼 제거
    if "frame" in df.columns:
        df = df.drop(columns=["frame"])

    x_np = df.to_numpy(dtype=np.float32)  # (T,F)

    # 길이 계산 (Zero padding 제거 로직)
    eps = 1e-8
    nonzero = np.any(np.abs(x_np) > eps, axis=1)
    if nonzero.any():
        length = int(np.max(np.where(nonzero)[0]) + 1)
    else:
        length = 1

    # Tensor 변환
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1,T,F)
    lengths = torch.tensor([length], dtype=torch.long).to(device)

    # 추론
    with torch.no_grad():
        logits = model(x, lengths)                # (1,2)
        probs = torch.softmax(logits, dim=1)[0]   # (2,)

    # 결과 정리
    p_normal = float(probs[0].cpu().item())
    p_abnormal = float(probs[1].cpu().item())
    pred_class = int(torch.argmax(probs).cpu().item()) # 0=normal, 1=abnormal

    return {
        "p_normal": p_normal,
        "p_abnormal": p_abnormal,
        "pred_class": pred_class
    }