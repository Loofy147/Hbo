SpokNAS — Rebuild v2

هذا المشروع هو إعادة بناء كاملة لمشروع SpokNAS مع تحسينات مصفوفية وعملية:

ميزات رئيسية:

* بنية NAS محمولة (islands, migration, speciation)
* تقييم متعدد الدقة (multifidelity) مع تنفيذ batched evaluation على PyTorch
* Vectorized utilities (pairwise distances, cosine similarity)
* Surrogate training vectorized وتجميع بيانات سريع
* GridAnalysis callback لحفظ صور وإحصاءات تلقائياً
* دعم لWandB وخرائط محفوظات التجارب
* متطلبات حديثة (transformers, peft, accelerate متاحة لاحقاً)

بدأ سريع:

1. إنشاء بيئة: `python -m venv .venv && source .venv/bin/activate` (Linux/Mac).
2. تثبيت المكتبات: `pip install -r requirements.txt`.
3. عدّل `config.yaml` ليتناسب مع جهازك.
4. ضع بياناتك في واجهة `data_manager` (يوجد placeholder في `trainer_main.py`).
5. شغّل بحث NAS: `python -m src.trainer_main --run_nas`.

ملاحظات أداء:

* استخدم GPU وPyTorch 2.x للحصول على تسريع ملحوظ.
* اضبط `batch_size` و`epochs_proxy` في `config.yaml` حسب الذاكرة.