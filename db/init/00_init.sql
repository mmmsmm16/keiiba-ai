-- Races table
CREATE TABLE IF NOT EXISTS races (
    race_id VARCHAR(20) PRIMARY KEY,
    date DATE NOT NULL,
    venue VARCHAR(50),
    race_number INTEGER,
    distance INTEGER,
    surface VARCHAR(20),
    weather VARCHAR(20),
    state VARCHAR(20),
    title VARCHAR(255)
);

-- Horses table
CREATE TABLE IF NOT EXISTS horses (
    horse_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100),
    sex VARCHAR(10),
    birthday DATE,
    sire_id VARCHAR(20),
    mare_id VARCHAR(20)
);

-- Results table
CREATE TABLE IF NOT EXISTS results (
    race_id VARCHAR(20),
    horse_id VARCHAR(20),
    jockey_id VARCHAR(20),
    trainer_id VARCHAR(20),
    frame_number INTEGER,
    horse_number INTEGER,
    rank INTEGER,
    time FLOAT,
    passing_rank VARCHAR(50),
    last_3f FLOAT,
    odds FLOAT,
    popularity INTEGER,
    weight INTEGER,
    weight_diff INTEGER,
    PRIMARY KEY (race_id, horse_id),
    FOREIGN KEY (race_id) REFERENCES races(race_id),
    FOREIGN KEY (horse_id) REFERENCES horses(horse_id)
);

-- Payouts table
CREATE TABLE IF NOT EXISTS payouts (
    race_id VARCHAR(20),
    ticket_type VARCHAR(20),
    winning_numbers VARCHAR(50),
    payout INTEGER,
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);
