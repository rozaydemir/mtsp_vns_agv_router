import pandas as pd
import os
from pathlib import Path

# Kaynak klasör (excel dosyalarının bulunduğu yer)
klasor_yolu = "excels/selector-model"  # burayı kendi klasör yoluna göre güncelle

# Sonuç dosyasının adı
output_dosya = "excels/last-review.xlsx"

# ExcelWriter ile yazacağız
with pd.ExcelWriter(output_dosya, engine="openpyxl") as writer:
    for dosya_adi in os.listdir(klasor_yolu):
        if dosya_adi.endswith(".xlsx"):
            tam_yol = os.path.join(klasor_yolu, dosya_adi)
            try:
                df = pd.read_excel(tam_yol, nrows=1002)
                sheet_adi = Path(dosya_adi).stem[:31]  # Excel'de sheet adı en fazla 31 karakter olabilir
                df.to_excel(writer, sheet_name=sheet_adi, index=False)
                print(f"{dosya_adi} -> {sheet_adi} eklendi.")
            except Exception as e:
                print(f"{dosya_adi} okunamadı: {e}")

print("bitti")
