import os
import pickle
import logging
import json
import pandas as pd
import numpy as np
from src.pipeline.config import ExperimentConfig
from src.model.lgbm import KeibaLGBM
from src.model.catboost_model import KeibaCatBoost
from src.model.tabnet_model import KeibaTabNet
from src.model.ensemble import EnsembleModel

logger = logging.getLogger(__name__)

def load_datasets(run_dir: str):
    dataset_path = os.path.join(run_dir, "data/lgbm_datasets.pkl")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)
    return datasets['train'], datasets['valid']

def train_lgbm(train_set, valid_set, params, run_dir):
    logger.info("⚡ Training LightGBM...")
    model = KeibaLGBM(params=params)
    model.train(train_set, valid_set)
    
    output_path = os.path.join(run_dir, "models/lgbm.pkl")
    model.save_model(output_path)
    model.plot_importance(os.path.join(run_dir, "reports/lgbm_importance.png"))
    return model

def train_catboost(train_set, valid_set, params, run_dir):
    logger.info("🐱 Training CatBoost...")
    model = KeibaCatBoost(params=params)
    model.train(train_set, valid_set)
    
    output_path = os.path.join(run_dir, "models/catboost.pkl")
    model.save_model(output_path)
    return model

def train_tabnet(train_set, valid_set, params, run_dir):
    logger.info("🕸️ Training TabNet...")
    
    # GPU競合対策: CUDA状態をリセット
    # CatBoost/LightGBMの後にPyTorchを使う場合、CUDAコンテキストが破損することがある
    import torch
    if torch.cuda.is_available():
        logger.info("🔄 CUDA状態をリセットしています...")
        torch.cuda.empty_cache()
        # cuBLASハンドルを強制的に再初期化
        try:
            # ダミー演算でcuBLASを初期化
            dummy = torch.randn(10, 10, device='cuda')
            _ = torch.matmul(dummy, dummy)
            del dummy
            torch.cuda.empty_cache()
            logger.info("✅ CUDA初期化成功")
        except Exception as e:
            logger.warning(f"⚠️ CUDA初期化失敗 - CPUにフォールバック: {e}")
            params = params.copy() if params else {}
            params['device_name'] = 'cpu'
    
    model = KeibaTabNet(params=params)
    model.train(train_set, valid_set)
    
    output_path = os.path.join(run_dir, "models/tabnet.zip")
    model.save_model(output_path)
    return model

def train_ensemble(train_set, valid_set, run_dir):
    logger.info("🤝 Training Ensemble Meta Model...")
    
    model = EnsembleModel()
    models_dir = os.path.join(run_dir, "models")
    
    # Check for base models. At least LGBM and CatBoost are expected for standard ensemble.
    # If TabNet exists, it will be loaded automatically by EnsembleModel if it looks for it.
    # Let's verify EnsembleModel behavior. Typically it loads lgbm.pkl, catboost.pkl, tabnet.zip if present.
    
    if not (os.path.exists(os.path.join(models_dir, "lgbm.pkl")) and 
            os.path.exists(os.path.join(models_dir, "catboost.pkl"))):
        raise RuntimeError("Base models (lgbm, catboost) missing for ensemble.")

    model.load_base_models(models_dir, version=None) 
    
    model.train_meta_model(valid_set)
    
    output_path = os.path.join(run_dir, "models/ensemble.pkl")
    model.save_model(output_path)
    return model

def train_roi_model(train_set, valid_set, params, run_dir, raw_data_path: str = None):
    """
    ROI最適化モデルの学習
    
    Args:
        train_set: {'X': DataFrame, 'y': Series, 'group': Array}
        valid_set: {'X': DataFrame, 'y': Series, 'group': Array}
        params: ROIモデルのハイパーパラメータ
        run_dir: 出力ディレクトリ
        raw_data_path: 生データ（odds, rank含む）へのパス
    """
    logger.info("💰 Training ROI Model...")
    
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from src.model.roi_model import ROIModel, RaceWinPredictor
    
    # デフォルトパラメータ
    params = params or {}
    model_type = params.get('model_type', 'simple')
    loss_type = params.get('loss_type', 'evmax')
    hidden_dim = params.get('hidden_dim', 128)
    num_layers = params.get('num_layers', 2)
    dropout = params.get('dropout', 0.3)
    epochs = params.get('epochs', 30)
    batch_size = params.get('batch_size', 64)
    lr = params.get('lr', 1e-3)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # 生データをロード（odds, rank情報が必要）
    if raw_data_path is None:
        raw_data_path = os.path.join(run_dir, "data/preprocessed_data.parquet")
    
    if not os.path.exists(raw_data_path):
        logger.warning(f"Raw data not found at {raw_data_path}. ROI model requires odds data.")
        return None
    
    raw_df = pd.read_parquet(raw_data_path)
    
    # 特徴量カラム
    feature_cols = train_set['X'].columns.tolist()
    
    # データ準備（レース単位でバッチ化）
    class RaceDataset(Dataset):
        def __init__(self, df, feature_cols, max_horses=18):
            self.max_horses = max_horses
            self.races = []
            
            # 数値型カラムのみを使用 (文字列カラムを除外)
            numeric_df = df[feature_cols].select_dtypes(include=[np.number])
            self.feature_cols = numeric_df.columns.tolist()
            logger.info(f"RaceDataset: Using {len(self.feature_cols)} numeric features (excluded non-numeric)")
            
            for race_id, grp in df.groupby('race_id'):
                if len(grp) < 3:
                    continue
                grp = grp.sort_values('horse_number')
                
                # 数値型カラムのみ取得
                X = grp[self.feature_cols].values.astype(np.float32)
                # NaN を 0 で埋める
                X = np.nan_to_num(X, nan=0.0)
                
                ranks = grp['rank'].values
                is_winner = (ranks == 1).astype(np.float32)
                odds = grp['odds'].fillna(1.0).values.astype(np.float32)
                
                self.races.append({
                    'X': X, 'is_winner': is_winner, 'odds': odds, 'n_horses': len(grp)
                })
        
        def __len__(self):
            return len(self.races)
        
        def __getitem__(self, idx):
            race = self.races[idx]
            n = min(race['n_horses'], self.max_horses)  # max_horsesでtruncate
            
            X_padded = np.zeros((self.max_horses, len(self.feature_cols)), dtype=np.float32)
            is_winner_padded = np.zeros(self.max_horses, dtype=np.float32)
            odds_padded = np.ones(self.max_horses, dtype=np.float32)
            mask = np.zeros(self.max_horses, dtype=np.float32)
            
            X_padded[:n] = race['X'][:n]  # max_horses分のみ使用
            is_winner_padded[:n] = race['is_winner'][:n]
            odds_padded[:n] = race['odds'][:n]
            mask[:n] = 1.0
            
            return {
                'X': torch.tensor(X_padded),
                'is_winner': torch.tensor(is_winner_padded),
                'odds': torch.tensor(odds_padded),
                'mask': torch.tensor(mask)
            }
    
    # カラム確認・補完
    for c in feature_cols:
        if c not in raw_df.columns:
            raw_df[c] = 0
    
    raw_df['rank'] = pd.to_numeric(raw_df['rank'], errors='coerce')
    raw_df['odds'] = pd.to_numeric(raw_df['odds'], errors='coerce').fillna(1.0)
    raw_df = raw_df.dropna(subset=['rank'])
    
    # Train/Valid分割
    train_years = raw_df['year'].min()  # 実際はconfigから取得すべき
    valid_year = raw_df['year'].max()
    train_df = raw_df[raw_df['year'] < valid_year].copy()
    valid_df = raw_df[raw_df['year'] == valid_year].copy()
    
    logger.info(f"Train: {len(train_df)} rows, Valid: {len(valid_df)} rows")
    
    train_dataset = RaceDataset(train_df, feature_cols)
    valid_dataset = RaceDataset(valid_df, feature_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    # モデル構築（RaceDatasetでフィルタリングされた特徴量数を使用）
    actual_feature_count = len(train_dataset.feature_cols)
    model = ROIModel(
        model_type=model_type, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers,
        dropout=dropout,
        device=device
    )
    model.build_model(actual_feature_count)
    
    # 損失関数
    from src.model.roi_loss import (
        EVMaxLoss, OddsWeightedBCE, ROIProxyLoss, 
        CombinedROILoss, AccuracyROILoss, RankingLoss
    )
    
    # 損失関数のパラメータ設定
    loss_params = params.get('loss_params', {})
    
    if loss_type == 'evmax':
        criterion = EVMaxLoss(**loss_params)
    elif loss_type == 'odds_bce':
        criterion = OddsWeightedBCE(**loss_params)
    elif loss_type == 'roi_proxy':
        criterion = ROIProxyLoss(**loss_params)
    elif loss_type == 'combined':
        criterion = CombinedROILoss(**loss_params)
    elif loss_type == 'accuracy_roi':
        # AccuracyROILoss(accuracy_weight=0.7, roi_weight=0.3)
        criterion = AccuracyROILoss(**loss_params)
    elif loss_type == 'ranking':
        criterion = RankingLoss(**loss_params)
    else:
        criterion = EVMaxLoss()  # デフォルト
    
    logger.info(f"Using loss function: {loss_type} ({type(criterion).__name__})")
    logger.info(f"Loss params: {loss_params}")
    
    # オプティマイザ
    optimizer = AdamW(model.model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 学習ループ
    best_roi = 0
    patience = params.get('patience', 20)  # デフォルト20 epoch
    patience_counter = 0
    
    logger.info(f"Early stopping patience: {patience}")
    
    for epoch in range(1, epochs + 1):
        # Train
        model.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            X = batch['X'].to(device)
            is_winner = batch['is_winner'].to(device)
            odds = batch['odds'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            probs = model.model(X, mask)
            loss = criterion(probs, is_winner, odds, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Eval
        model.model.eval()
        total_cost = 0
        total_return = 0
        n_hits = 0
        n_races = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                X = batch['X'].to(device)
                is_winner = batch['is_winner'].to(device)
                odds = batch['odds'].to(device)
                mask = batch['mask'].to(device)
                
                probs = model.model(X, mask)
                
                batch_size_cur = X.shape[0]
                for i in range(batch_size_cur):
                    valid_mask = mask[i].bool()
                    valid_probs = probs[i][valid_mask]
                    valid_winner = is_winner[i][valid_mask]
                    valid_odds = odds[i][valid_mask]
                    
                    if len(valid_probs) == 0:
                        continue
                    
                    top1_idx = valid_probs.argmax()
                    is_hit = valid_winner[top1_idx].item() == 1
                    
                    total_cost += 100
                    if is_hit:
                        total_return += valid_odds[top1_idx].item() * 100
                        n_hits += 1
                    n_races += 1
        
        roi = (total_return / total_cost * 100) if total_cost > 0 else 0
        acc = (n_hits / n_races * 100) if n_races > 0 else 0
        
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d} | Loss: {total_loss/n_batches:.4f} | ROI: {roi:.1f}% | Acc: {acc:.1f}%")
        
        if roi > best_roi:
            best_roi = roi
            patience_counter = 0  # リセット
            model.save(os.path.join(run_dir, 'models', 'roi_model_best.pt'))
            logger.info(f"New best ROI! Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (Best ROI: {best_roi:.1f}%)")
                break
    
    logger.info(f"✅ ROI Model Training Completed. Best ROI: {best_roi:.1f}%")
    return model

def train_model(config: ExperimentConfig, run_dir: str):
    train_set, valid_set = load_datasets(run_dir)
    
    # 特徴量削除 (Config制御)
    if config.data.drop_features:
        logger.info(f"指定された特徴量を削除します: {config.data.drop_features}")
        for ds in [train_set, valid_set]:
            if 'X' in ds and isinstance(ds['X'], pd.DataFrame):
                # 存在しないカラムは無視 (errors='ignore')
                ds['X'] = ds['X'].drop(columns=config.data.drop_features, errors='ignore')
    
    model_type = config.model.type
    
    trained_models = {}
    
    # ensemble_only: ベースモデル学習をスキップし、メタモデルのみ学習
    run_lgbm = (model_type == 'lgbm' or model_type == 'ensemble')
    run_catboost = (model_type == 'catboost' or model_type == 'ensemble')
    
    # TabNet実行判定: tabnet単体指定 or ensemble指定
    # tabnet_params.enabled == false の場合はスキップ
    tabnet_enabled = True
    if config.model.tabnet_params and config.model.tabnet_params.get('enabled') is False:
        tabnet_enabled = False
        logger.info("⏭️ TabNetはskipされます (enabled: false)")
    
    run_tabnet = (model_type == 'tabnet' and tabnet_enabled)
    if model_type == 'ensemble' and tabnet_enabled:
         run_tabnet = True

    run_ensemble = (model_type == 'ensemble' or model_type == 'ensemble_only')
    
    # ensemble_only の場合、ベースモデル学習をスキップ（ただし、存在しない有効なモデルは学習する）
    if model_type == 'ensemble_only':
        models_dir = os.path.join(run_dir, "models")
        logger.info("🚀 ensemble_only モード: 既存モデルを確認し、不足分のみ学習します")
        
        # LightGBM: 常に必要
        if os.path.exists(os.path.join(models_dir, "lgbm.pkl")):
            run_lgbm = False
            logger.info("  ✅ LightGBM: 既存モデルを使用")
        else:
            run_lgbm = True
            logger.info("  🔧 LightGBM: モデルが見つかりません - 学習します")
        
        # CatBoost: 常に必要
        if os.path.exists(os.path.join(models_dir, "catboost.pkl")):
            run_catboost = False
            logger.info("  ✅ CatBoost: 既存モデルを使用")
        else:
            run_catboost = True
            logger.info("  🔧 CatBoost: モデルが見つかりません - 学習します")
        
        # TabNet: enabled=trueの場合のみ確認
        if tabnet_enabled:
            if os.path.exists(os.path.join(models_dir, "tabnet.zip")):
                run_tabnet = False
                logger.info("  ✅ TabNet: 既存モデルを使用")
            else:
                run_tabnet = True
                logger.info("  🔧 TabNet: モデルが見つかりません - 学習します")
        else:
            run_tabnet = False
            logger.info("  ⏭️ TabNet: 無効化されています")

    if run_lgbm:
        lgbm = train_lgbm(train_set, valid_set, config.model.lgbm_params, run_dir)
        trained_models['lgbm'] = lgbm
        
    if run_catboost:
        cat = train_catboost(train_set, valid_set, config.model.catboost_params, run_dir)
        trained_models['catboost'] = cat
        
    if run_tabnet:
        # TabNet params default if None
        t_params = config.model.tabnet_params if config.model.tabnet_params else {}
        tab = train_tabnet(train_set, valid_set, t_params, run_dir)
        trained_models['tabnet'] = tab
        
    if run_ensemble:
        ens = train_ensemble(train_set, valid_set, run_dir)
        trained_models['ensemble'] = ens
    
    # ROIモデル
    if model_type == 'roi':
        logger.info("💰 ROI Model Training Mode")
        roi_model = train_roi_model(train_set, valid_set, config.model.roi_params, run_dir)
        trained_models['roi'] = roi_model
        
    logger.info("✅ Model Training Completed.")
