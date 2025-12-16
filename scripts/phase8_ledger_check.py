"""
Phase 8 Ledger Consistency Check

Compare tickets purchased in Final vs Snapshot conditions.
Generate hash for proof of (dis)similarity.
"""
import pandas as pd
import hashlib

def compute_ledger_hash(df, cols=['race_id', 'horse_number']):
    """Compute MD5 hash of sorted ticket list"""
    # Sort and create string representation
    sorted_df = df[cols].sort_values(cols)
    str_repr = sorted_df.to_csv(index=False)
    return hashlib.md5(str_repr.encode()).hexdigest()

def main():
    print("=" * 70)
    print("LEDGER CONSISTENCY CHECK")
    print("=" * 70)
    
    # Load ledgers
    ledger_final = pd.read_parquet('reports/phase8_snapshot/win_ev/ledger_final.parquet')
    ledger_snap = pd.read_parquet('reports/phase8_snapshot/win_ev/ledger_snapshot.parquet')
    
    print(f"\nFinal Ledger: {len(ledger_final)} bets")
    print(f"Snapshot Ledger: {len(ledger_snap)} bets")
    
    # Create ticket key (race_id + horse_number)
    ledger_final['ticket_key'] = ledger_final['race_id'].astype(str) + '_' + ledger_final['horse_number'].astype(str)
    ledger_snap['ticket_key'] = ledger_snap['race_id'].astype(str) + '_' + ledger_snap['horse_number'].astype(str)
    
    final_tickets = set(ledger_final['ticket_key'])
    snap_tickets = set(ledger_snap['ticket_key'])
    
    # Set comparison
    both = final_tickets & snap_tickets
    only_final = final_tickets - snap_tickets
    only_snap = snap_tickets - final_tickets
    
    print(f"\nTicket Overlap: {len(both)}")
    print(f"Only in Final: {len(only_final)}")
    print(f"Only in Snapshot: {len(only_snap)}")
    
    overlap_pct = len(both) / len(final_tickets | snap_tickets) * 100
    print(f"Jaccard Similarity: {overlap_pct:.1f}%")
    
    # Compute hash
    hash_final = compute_ledger_hash(ledger_final)
    hash_snap = compute_ledger_hash(ledger_snap)
    
    print(f"\nFinal Ledger Hash: {hash_final}")
    print(f"Snapshot Ledger Hash: {hash_snap}")
    print(f"Hashes Match: {hash_final == hash_snap}")
    
    # Sample differences
    if only_final:
        print(f"\nSample tickets ONLY in Final (dropped in Snapshot):")
        for tk in list(only_final)[:5]:
            row = ledger_final[ledger_final['ticket_key'] == tk].iloc[0]
            print(f"  {tk}: prob={row['prob']:.4f}, odds={row['odds']:.1f}, ev={row['ev']:.3f}")
    
    if only_snap:
        print(f"\nSample tickets ONLY in Snapshot (added vs Final):")
        for tk in list(only_snap)[:5]:
            row = ledger_snap[ledger_snap['ticket_key'] == tk].iloc[0]
            print(f"  {tk}: prob={row['prob']:.4f}, odds={row['odds']:.1f}, ev={row['ev']:.3f}")
    
    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"The two ledgers are {'IDENTICAL' if hash_final == hash_snap else 'DIFFERENT'}.")
    print(f"Overlap: {len(both)} / {len(final_tickets | snap_tickets)} = {overlap_pct:.1f}%")
    
if __name__ == '__main__':
    main()
