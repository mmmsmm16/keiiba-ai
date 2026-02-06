# Phase 12: Ticket Price Availability Matrix

## 1. Data Source Status
Based on database inspection of `apd_sokuho_*` tables:

| Table | Content | Min Year | Max Year | Status |
|---|---|---|---|---|
| `apd_sokuho_o1` | Win, Place, Wakuren | 2014 | 2025 | **Available** |
| `apd_sokuho_o2` | Umaren | 2014 | 2025 | **Available** |
| `apd_sokuho_o3` | Wide | - | - | **Unavailable** (Empty) |
| `apd_sokuho_o4` | Umatan | - | - | **Unavailable** (Empty) |
| `apd_sokuho_o5` | Sanrenpuku | - | - | **Unavailable** (Empty) |
| `apd_sokuho_o6` | Sanrentan | - | - | **Unavailable** (Empty) |

## 2. Evaluation Mode by Ticket Type

| Ticket Type | EV Mode | History (2014-2024) | 2025 | Notes |
|---|---|---|---|---|
| **Win** | Price Aware | ✅ Available | ✅ Available | Strict EV simulation possible |
| **Place** | Price Aware | ✅ Available | ✅ Available | Strict EV simulation possible |
| **Umaren** | Price Aware | ✅ Available | ✅ Available | Strict EV simulation possible |
| **Wakuren** | Price Aware | ✅ Available | ✅ Available | Strict EV simulation possible |
| **Wide** | Price Agnostic | ❌ No Odds | ❌ No Odds | Sim only (using Final Odds?) |
| **Umatan** | Price Agnostic | ❌ No Odds | ❌ No Odds | Sim only (using Final Odds?) |
| **Sanrenpuku** | Price Agnostic | ❌ No Odds | ❌ No Odds | Sim only (using Final Odds?) |
| **Sanrentan** | Price Agnostic | ❌ No Odds | ❌ No Odds | Sim only (using Final Odds?) |

## 3. Impact on Phase 12
- **Price Aware Portfolio**: Can only reliably optimize allocation between Win, Place, Umaren, Wakuren using real-time odds logic.
- **Price Agnostic Exploration**: Other tickets can be generated and evaluated using *Final Odds* as a proxy, but this introduces "Lookahead Bias" regarding price availability (we assume we could buy at that price, though actual odds might differ).
- **Recommendation**:
  - Focus strict Nested WF optimization on **Win/Place/Umaren**.
  - Develop candidate generation for all tickets (Task 3), but flag EV calculation as "Provisional" for unconnected tickets.
