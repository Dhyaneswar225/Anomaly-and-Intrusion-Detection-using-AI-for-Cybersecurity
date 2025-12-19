import pandas as pd

# ================= CONFIG =================
INPUT_TXT = "F://Master Thesis//anomaly-ids//data//raw//nsl-kdd//KDDTest+.txt"     # your raw text file
OUTPUT_CSV = "F://Master Thesis//anomaly-ids//data//processed//KDDTest+.csv"    # output CSV

COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files',
    'num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','label','difficulty'
]

# ================= LOAD RAW TEXT =================
rows = []
with open(INPUT_TXT, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        values = line.split(",")

        if len(values) != len(COLUMNS):
            raise ValueError(
                f"Row has {len(values)} columns, expected {len(COLUMNS)}:\n{line}"
            )

        rows.append(values)

# ================= CREATE DATAFRAME =================
df = pd.DataFrame(rows, columns=COLUMNS)

# Convert numeric columns automatically
for col in COLUMNS:
    if col not in ["protocol_type", "service", "flag", "label"]:
        df[col] = pd.to_numeric(df[col], errors="ignore")

# ================= SAVE CSV =================
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Converted {len(df)} rows → {OUTPUT_CSV}")
print(df.head())
