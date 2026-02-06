
sample = "01020002810080103000125005010400028900901050004010110106000588013"

def parse_umaren(s):
    # Try 13 char chunk
    chunk_size = 15 # Hypothesis: H1(2)+H2(2)+Odds(6)+Pop(3) = 13? Or more?
    # 0102 000281 008 (13?)
    # 0103 000125 005 (13?)
    
    # Let's print chunks of 13
    print("--- 13 Char Chunks ---")
    for i in range(0, len(s), 13):
        print(s[i:i+13])
        
    # Let's try to see if it matches
    # Chunk 1: 0102000281008 (H1:01 H2:02 O:28.1 P:8)
    # Chunk 2: 0103000125005 (H1:01 H3:03 O:12.5 P:5)
    
if __name__ == "__main__":
    parse_umaren(sample)
