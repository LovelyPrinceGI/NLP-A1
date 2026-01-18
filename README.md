โค้ด `app.py` เวอร์ชันนี้ **ครบและสอดคล้องกับ search_backend แบบ B แล้ว** ไม่ต้องแก้อะไรเพิ่มครับ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/108109695/b7df8e3d-3371-4ec1-873b-747bd09f18a8/search_backend.py)

สิ่งที่ตรงโจทย์แล้ว:

- import `search_topk` ที่รับ `model_name` ✔  
- ฟอร์มมี:
  - ช่อง text `name="q"` รับ query ✔  
  - `<select name="model">` ให้เลือก `glove` หรือ `skipgram_ns` และใช้ `selected` ตามค่าปัจจุบัน ✔  
- route `/`:
  - อ่าน `q` และ `model` จาก query string  
  - ถ้ามี query → เรียก `search_topk(query, k=10, model_name=model_name)` ✔  
  - ส่ง `query`, `results`, `model` เข้า template ✔  
- แสดงผล:
  - หัวข้อ `Results for "query" (model: model)` ✔  
  - loop `results` แสดง `r.text` และ `score` แบบ format 4 ทศนิยม ✔  

จากนี้ถ้าไฟล์อื่นพร้อมแล้ว (`models.py`, `search_backend.py`, `prepare_context_embeddings.py`, `.pt`, `.pkl`, `.npy`, `.txt`) ขั้นตอนรันคือ:

```bash
cd "C:\Users\Legion 5 Pro\OneDrive\Documents\NLP\assignment\A1"

# ถ้ายังไม่รันเตรียม context
python prepare_context_embeddings.py

# รันเว็บ
python app.py
```

แล้วเปิด `http://127.0.0.1:5000/` พิมพ์ query + เลือกโมเดลได้เลยครับ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/108109695/b0348039-57b4-4cef-839e-8849e2cdf338/app.py)