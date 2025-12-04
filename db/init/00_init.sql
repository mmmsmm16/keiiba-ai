-- Races テーブル（レース情報）
CREATE TABLE IF NOT EXISTS races (
    race_id VARCHAR(20) PRIMARY KEY,
    date DATE NOT NULL,
    venue VARCHAR(50), -- 開催場所
    race_number INTEGER, -- レース番号
    distance INTEGER, -- 距離
    surface VARCHAR(20), -- 馬場 (芝, ダート)
    weather VARCHAR(20), -- 天候
    state VARCHAR(20), -- 馬場状態
    title VARCHAR(255) -- レース名
);

-- Horses テーブル（馬基本情報）
CREATE TABLE IF NOT EXISTS horses (
    horse_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100), -- 馬名
    sex VARCHAR(10), -- 性別
    birthday DATE, -- 生年月日
    sire_id VARCHAR(20), -- 父ID
    mare_id VARCHAR(20) -- 母ID
);

-- Results テーブル（レース結果）
CREATE TABLE IF NOT EXISTS results (
    race_id VARCHAR(20),
    horse_id VARCHAR(20),
    jockey_id VARCHAR(20), -- 騎手ID
    trainer_id VARCHAR(20), -- 調教師ID
    frame_number INTEGER, -- 枠番
    horse_number INTEGER, -- 馬番
    rank INTEGER, -- 着順
    time FLOAT, -- タイム
    passing_rank VARCHAR(50), -- 通過順
    last_3f FLOAT, -- 上がり3F
    odds FLOAT, -- 単勝オッズ
    popularity INTEGER, -- 人気
    weight INTEGER, -- 馬体重
    weight_diff INTEGER, -- 体重増減
    age INTEGER, -- 年齢
    PRIMARY KEY (race_id, horse_id),
    FOREIGN KEY (race_id) REFERENCES races(race_id),
    FOREIGN KEY (horse_id) REFERENCES horses(horse_id)
);

-- Payouts テーブル（払い戻し）
CREATE TABLE IF NOT EXISTS payouts (
    race_id VARCHAR(20),
    ticket_type VARCHAR(20), -- 券種
    winning_numbers VARCHAR(50), -- 当選馬番
    payout INTEGER, -- 払戻金
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);
