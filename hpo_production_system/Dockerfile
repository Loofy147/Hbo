# Dockerfile - للحاويات

# ابدأ من صورة Python رسمية
FROM python:3.9-slim

# تعيين مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ ملفات المتطلبات أولاً للاستفادة من التخزين المؤقت لـ Docker
COPY requirements.txt .

# تثبيت المتطلبات
# --no-cache-dir لتقليل حجم الصورة
RUN pip install --no-cache-dir -r requirements.txt

# نسخ جميع ملفات المشروع إلى مجلد العمل
COPY . .

# تعيين نقطة الدخول الافتراضية
# سيتم تشغيل هذا الأمر عند بدء تشغيل الحاوية
# مثال: docker run hpo-system --quick
ENTRYPOINT ["python", "run_hpo.py"]

# الأمر الافتراضي إذا لم يتم توفير أي وسيطات
CMD ["--help"]