# Selective Multi-Pass Results

| System | Precision | Recall | F0.5 |
|---|---:|---:|---:|
| pass1 | 0.2143 | 0.0728 | 0.1543 |
| pass2 | 0.0913 | 0.0922 | 0.0915 |
| pass3 | 0.0931 | 0.0922 | 0.0930 |
| stable | 0.0936 | 0.0922 | 0.0933 |
| selective | 0.1974 | 0.0728 | 0.1471 |
| distance_oracle | 0.1782 | 0.0874 | 0.1475 |

## Analysis

- Test Examples: 200
- Pass1 Equals Pass2: 69
- Pass2 Equals Pass3: 190
- Second Pass Better By Token Distance: 23
- Second Pass Worse By Token Distance: 4
- Selector Chose Pass2: 6
- Average Stable Passes: 1.7
