import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.dataset import RCDataset

class LogStandardScaler:
    """
    回路パラメータ (R, C, ...) に対する対数変換と Z-score 正規化を行うクラス。
    
    入力形式: (N, num_features) の Tensor. 
    仮定: 0 列目が R, 1 列目が C, ... (拡張可能)
    
    処理フロー:
    1. 入力：物理値 [R, C] (またはそれ以上の次元)
    2. 対数変換：log10(各パラメータ) を計算
    3. 正規化：StandardScaler で平均 0, 分散 1 に変換
    
    逆変換フロー:
    1. 入力：正規化されたデータ (N, num_features)
    2. 標準化の逆変換：log10 空間の値を復元
    3. 指数変換：10^x を計算して元の物理値 [R, C] を復元
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _validate_features(self, x: torch.Tensor) -> torch.Tensor:
        """形状の検証と整列 (N,) -> (N, 1), (N, num_features) 保証"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        if x.dim() == 1:
            x = x.unsqueeze(1)
        elif x.dim() != 2:
            raise ValueError(f"Input must be 1D or 2D tensor, got {x.dim()}D")
            
        return x

    def fit(self, parameters: torch.Tensor):
        """
        訓練データから統計量を学習する。
        
        Args:
            parameters: 回路パラメータ (N, num_features). 
                        例: [R, C] の順に結合した tensor.
        """
        x = self._validate_features(parameters)
        
        # 対数変換 (各列ごとに独立に計算)
        log_x = torch.log10(x)
        
        # numpy 変換して sklearn に渡す
        inputs_log = log_x.numpy()
        
        self.scaler.fit(inputs_log)
        self.is_fitted = True
        return self

    def transform(self, parameters: torch.Tensor) -> np.ndarray:
        """
        対数変換と標準化を実行する。
        
        Args:
            parameters: (N, num_features)
            
        Returns:
            標準化されたデータ (N, num_features) numpy array
        """
        if not self.is_fitted:
            raise RuntimeError("LogStandardScaler is not fitted. Call fit() first.")
            
        x = self._validate_features(parameters)
        log_x = torch.log10(x)
        
        inputs_log = log_x.numpy()
        
        return self.scaler.transform(inputs_log)

    def fit_transform(self, parameters: torch.Tensor) -> np.ndarray:
        """fit と transform をまとめて実行する。"""
        self.fit(parameters)
        return self.transform(parameters)

    def inverse_transform(self, scaled_data: np.ndarray) -> torch.Tensor:
        """
        標準化されたデータを元の物理値に戻す。
        
        Args:
            scaled_data: 標準化されたデータ (N, num_features)
            
        Returns:
            復元されたパラメータ (N, num_features) torch.Tensor
            [R, C, ...] の順で復元されます
        """
        if not self.is_fitted:
            raise RuntimeError("LogStandardScaler is not fitted. Call fit() first.")
            
        # 1. StandardScaler で log 空間の値を復元
        log_inputs = self.scaler.inverse_transform(scaled_data)
        
        # 2. 指数関数で元の物理値に戻す
        # numpy 配列のまま 10^x を計算し、Tensor に変換
        recovered_params = torch.tensor(10 ** log_inputs, dtype=torch.float32)
        
        return recovered_params


class HScaler:
    """
    周波数応答 H をスケーリングするクラス。
    
    処理フロー:
    1. H (複素数) を絶対値 (mags) と位相 (angles) に分解
    2. 絶対値のみ log10 変換 (epsilon で 0 除算防止) 後、StandardScaler で標準化
    3. 位相はそのまま保持
    4. transform 出力: [標準化された mags, angles] を横に結合した (N, 2 * num_freqs)
    
    逆変換フロー:
    1. 入力 (N, 2 * num_freqs) を前半 (mags) と後半 (angles) に分割
    2. 標準化された mags を逆変換して log 空間を復元 -> 10^x で元の絶対値を復元
    3. 復元した mags と angles から複素数 H を再構築
    """
    def __init__(self, epsilon: float = 1e-9):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.epsilon = epsilon
        self.num_freqs = None  # 学習時に記録

    def _validate_Hs(self, Hs: torch.Tensor) -> torch.Tensor:
        """入力形状の検証とnumpy変換 (N, num_freqs) を保証"""
        if not isinstance(Hs, torch.Tensor):
            Hs = torch.tensor(Hs, dtype=torch.complex64)
        
        if Hs.dim() != 2:
            raise ValueError(f"Hs must be 2D tensor (N, num_freqs), got {Hs.dim()}D")
            
        return Hs

    def fit(self, Hs: torch.Tensor) -> 'HScaler':
        """
        訓練データから絶対値の統計量を学習する。
        
        Args:
            Hs: 周波数応答 (N, num_freqs) の複素数テンソル
        """
        Hs = self._validate_Hs(Hs)
        self.num_freqs = Hs.shape[1]
        
        # 絶対値のみ抽出
        mags = torch.abs(Hs)  # (N, num_freqs)
        
        # 対数変換 (epsilon を加えて 0 未満を防ぐ)
        log_mags = torch.log10(mags + self.epsilon)
        
        # numpy 変換して sklearn に渡す
        inputs_log = log_mags.numpy()
        
        self.scaler.fit(inputs_log)
        self.is_fitted = True
        return self

    def transform(self, Hs: torch.Tensor) -> np.ndarray:
        """
        絶対値の標準化と位相の抽出、結合を実行する。
        
        Args:
            Hs: 周波数応答 (N, num_freqs)
            
        Returns:
            結合された特徴量ベクトル (N, 2 * num_freqs)
            前半 [0:num_freqs]: 標準化された絶対値
            後半 [num_freqs:2*num_freqs]: 位相 (ラジアン)
        """
        if not self.is_fitted:
            raise RuntimeError("HScaler is not fitted. Call fit() first.")
            
        Hs = self._validate_Hs(Hs)
        
        # 絶対値
        mags = torch.abs(Hs)
        log_mags = torch.log10(mags + self.epsilon)
        inputs_log = log_mags.numpy()
        
        # 標準化
        scaled_mags = self.scaler.transform(inputs_log)
        
        # 位相
        angles = torch.angle(Hs).numpy()
        
        # 結合: (N, num_freqs) + (N, num_freqs) -> (N, 2 * num_freqs)
        # 前半: mags, 後半: angles
        combined = np.concatenate([scaled_mags, angles], axis=1)
        
        return combined

    def fit_transform(self, Hs: torch.Tensor) -> np.ndarray:
        """fit と transform をまとめて実行する。"""
        self.fit(Hs)
        return self.transform(Hs)

    def inverse_transform(self, combined_data: np.ndarray) -> torch.Tensor:
        """
        結合された特徴量から元の複素数 H を復元する。
        
        Args:
            combined_data: (N, 2 * num_freqs)
                前半: 標準化された絶対値
                後半: 位相
            
        Returns:
            復元された周波数応答 (N, num_freqs) torch.Tensor (complex)
        """
        if not self.is_fitted:
            raise RuntimeError("HScaler is not fitted. Call fit() first.")
            
        if combined_data.shape[1] != 2 * self.num_freqs:
            raise ValueError(f"Input dimension mismatch. Expected {2 * self.num_freqs}, got {combined_data.shape[1]}")
        
        # 分割
        scaled_mags = combined_data[:, :self.num_freqs]
        angles = combined_data[:, self.num_freqs:]
        
        # 絶対値の逆変換 (標準化 -> log 空間)
        log_mags = self.scaler.inverse_transform(scaled_mags)
        
        # 指数関数で元の絶対値を復元 (epsilon を引くか、加算前の状態に戻す)
        # log10(mags + eps) = log_mags => mags + eps = 10^log_mags => mags = 10^log_mags - eps
        # ただし、通常は EPS が非常に小さいため、10^log_mags だけで十分な近似となる。
        # 厳密さを求めるなら 10^log_mags - epsilon だが、10^log_mags で十分とみなす。
        mags = 10 ** log_mags
        
        # 複素数再構築
        # mags: (N, num_freqs), angles: (N, num_freqs)
        # torch.view_as_complex は最後の次元が [real, imag] の形式を期待するため、
        # 自分で計算する方が簡単。
        # H = mag * exp(j * angle) = mag * (cos(angle) + j * sin(angle))
        
        mags_t = torch.tensor(mags, dtype=torch.float32)
        angles_t = torch.tensor(angles, dtype=torch.float32)
        
        real = mags_t * torch.cos(angles_t)
        imag = mags_t * torch.sin(angles_t)
        
        Hs_reconstructed = torch.view_as_complex(torch.stack([real, imag], dim=-1))
        
        return Hs_reconstructed

if __name__ == "__main__":
    print("--- 新インターフェースの動作確認 ---")
    fs = np.logspace(1, 5, 100)
    dataset = RCDataset(num_samples=1000, R_range=[10e3, 100e3], C_range=[1e-9, 10e-9], fs=fs)
    
    # データ抽出
    H_all, circuit_paramters_all = dataset[:]
    R_all = circuit_paramters_all[:, 0]
    C_all = circuit_paramters_all[:, 1]
    
    # 1. LogStandardScaler の初期化と fit
    scaler = LogStandardScaler()
    print(f"Fitting scaler on {circuit_paramters_all.shape[0]} samples (features: {circuit_paramters_all.shape[1]})...")
    scaler.fit(circuit_paramters_all)
    
    # 2. Transform (正規化)
    X_scaled = scaler.transform(circuit_paramters_all)
    print(f"Scaled data shape: {X_scaled.shape}")
    print(f"Mean of scaled R (col 0): {np.mean(X_scaled[:, 0]):.4f}, Std: {np.std(X_scaled[:, 0]):.4f}")
    print(f"Mean of scaled C (col 1): {np.mean(X_scaled[:, 1]):.4f}, Std: {np.std(X_scaled[:, 1]):.4f}")
    
    # 3. Inverse Transform (元に戻す)
    recovered_params = scaler.inverse_transform(X_scaled)
    
    # 誤差確認 (recovered_params は (N, 2) の Tensor)
    R_recovered = recovered_params[:, 0]
    C_recovered = recovered_params[:, 1]
    
    max_error_R = torch.max(torch.abs(R_all - R_recovered))
    max_error_C = torch.max(torch.abs(C_all - C_recovered))
    
    print(f"\nRecovery Check:")
    print(f"Max Error in R: {max_error_R:.2e} (Original: {R_all[0]:.2f} -> Recovered: {R_recovered[0]:.2f})")
    print(f"Max Error in C: {max_error_C:.2e} (Original: {C_all[0]:.2e} -> Recovered: {C_recovered[0]:.2e})")
    
    # 統合版のメリット確認：直接比較
    max_error_total = torch.max(torch.abs(circuit_paramters_all - recovered_params))
    print(f"Max Total Parameter Error: {max_error_total:.2e}")

    if max_error_R < 1e-1 and max_error_C < 1e-8:
        print("✓ Success: Inverse transform correctly recovered original values.")
    else:
        print("✗ Error: Values did not match.")

    # --- HScaler テスト ---
    print("\n--- HScaler 動作確認 ---")
    h_scaler = HScaler()
    h_scaler.fit(H_all)
    
    # Transform
    H_combined = h_scaler.transform(H_all)
    print(f"H combined shape: {H_combined.shape}") # 期待: (N, 200)
    
    # Inverse Transform
    H_reconstructed = h_scaler.inverse_transform(H_combined)
    
    # 誤差確認
    max_mag_error = torch.max(torch.abs(torch.abs(H_all) - torch.abs(H_reconstructed)))
    max_angle_error = torch.max(torch.abs(torch.angle(H_all) - torch.angle(H_reconstructed)))
    max_complex_error = torch.max(torch.abs(H_all - H_reconstructed))
    
    print(f"Max Mag Error: {max_mag_error:.2e}")
    print(f"Max Angle Error: {max_angle_error:.2e} rad")
    print(f"Max Complex Error: {max_complex_error:.2e}")
    
    if max_complex_error < 1e-5:
        print("✓ Success: HScaler correctly recovered H values.")
    else:
        print("✗ Error: H values did not match.")
