import pandas as pd
import json
import re

# Verilen Excel dosyasını yükleme
file_path = 'result_all.xlsx'
df = pd.read_excel(file_path, skiprows=1)

# Başlıkları düzeltme
df.columns = ["TEST ID", "FILE NAME", "VEHICLE COUNT", "TROLLEY COUNT", "TROLLEY IMPACT TIME", "EARLINESS/TARDINESS PENALTY",
              "Math Form Is Optimal OR Feasible", "Math CPU TIME", "Math RESULT", "Math Model Distance Cost",
              "Math Model EA Cost", "Math Model TA Cost", "Math ROUTES", "ALNS CPU TIME", "ALNS RESULT",
              "ALNS Distance Cost", "ALNS EA Cost", "ALNS TA Cost", "ALNS ROUTES", "ROTH CPU TIME", "ROTH RESULT",
              "ROTH Distance Cost", "ROTH EA Cost", "ROTH TA Cost", "ROTH ROUTES",
              "GAP"]
df = df.drop(columns=["Math ROUTES", "ALNS ROUTES", "ROTH ROUTES"])
def clean_key(key):
    # Özel karakterleri silme ve boşlukları "_" ile değiştirme
    key = re.sub(r'[^A-Za-z0-9\s]', '', key)
    key = key.replace(" ", "_")
    return key


df.columns = [clean_key(col) for col in df.columns]
# Başlık satırından sonraki tüm verileri alma
data = df.to_dict(orient='records')





# JSON verisini dosyaya yazma
json_file_path = 'lrc15.json'
with open('lrc15.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

# JSON dosyasını kullanıcıya döndürme
json_file_path
