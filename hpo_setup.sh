#!/bin/bash
# hpo_setup.sh - إعداد المشروع بالكامل

echo "🚀 بدء إعداد نظام HPO للإنتاج..."

# اسم المجلد الذي سيتم إنشاؤه
DEST_DIR="hpo_production_system"

# التحقق مما إذا كان المجلد موجودًا بالفعل
if [ -d "$DEST_DIR" ]; then
    echo "⚠️ تحذير: المجلد '$DEST_DIR' موجود بالفعل."
    read -p "هل تريد الكتابة فوقه؟ (y/n): " overwrite
    if [ "$overwrite" != "y" ]; then
        echo "🛑 تم إلغاء الإعداد."
        exit 1
    fi
    rm -rf "$DEST_DIR"
fi

# إنشاء المجلد الرئيسي
mkdir -p "$DEST_DIR"
echo "✅ تم إنشاء المجلد '$DEST_DIR'."

echo "📂 نسخ ملفات المشروع..."

# قائمة الملفات والمجلدات لنسخها
FILES_TO_COPY=(
    "hpo_production_system.py"
    "run_hpo.py"
    "quick_start.py"
    "install.py"
    "requirements.txt"
    "setup.py"
    "start.sh"
    "start.bat"
    "examples"
    "configs"
    "README.md"
    "QUICK_START.md"
    "Dockerfile"
    "docker-compose.yml"
)

# نسخ الملفات والمجلدات
for item in "${FILES_TO_COPY[@]}"; do
    if [ -e "$item" ]; then
        cp -r "$item" "$DEST_DIR/"
        echo "  - تم نسخ $item"
    else
        echo "  - ⚠️ تحذير: الملف أو المجلد '$item' غير موجود، تم تخطيه."
    fi
done

echo "✅ تم إعداد المشروع بنجاح!"
echo ""
echo "👇 الخطوات التالية:"
echo "1. انتقل إلى المجلد الجديد:"
echo "   cd $DEST_DIR"
echo ""
echo "2. قم بتثبيت المتطلبات:"
echo "   python install.py"
echo ""
echo "3. قم بتشغيل مثال سريع:"
echo "   python run_hpo.py --quick"
echo ""
echo "🎉 استمتع بتحسين معاملاتك!"