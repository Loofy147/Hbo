import optuna
import os
import datetime
import yaml
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

class HPOSystem:
    """
    نظام متكامل لتحسين المعاملات الفائقة (HPO) جاهز للإنتاج.
    """
    def __init__(
        self,
        search_space,
        objective_function,
        n_trials=100,
        direction="maximize",
        sampler="TPE",
        pruner=None,
        n_jobs=1,
        study_name=None,
        storage_path="hpo_studies.db",
        config_path="configs/default.yaml"
    ):
        """
        تهيئة نظام HPO.

        Args:
            search_space (dict): قاموس يصف مساحة البحث.
            objective_function (callable): دالة الهدف التي سيتم تحسينها.
            n_trials (int): عدد التجارب لتشغيلها.
            direction (str): اتجاه التحسين ('maximize' or 'minimize').
            sampler (str): نوع الـ sampler ('TPE', 'GP', 'Random').
            pruner (str or None): نوع الـ pruner ('ASHA' or None).
            n_jobs (int): عدد المهام المتوازية.
            study_name (str or None): اسم الدراسة. إذا لم يتم توفيره، يتم إنشاء اسم تلقائي.
            storage_path (str): مسار ملف قاعدة البيانات لتخزين النتائج.
            config_path (str): مسار ملف الإعدادات الافتراضية.
        """
        self.search_space = search_space
        self.objective_function = objective_function
        self.n_trials = n_trials
        self.direction = direction
        self.n_jobs = n_jobs

        # تحميل الإعدادات من ملف YAML
        self.config = self._load_config(config_path)

        self.sampler = self._get_sampler(sampler)
        self.pruner = self._get_pruner(pruner)

        if study_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.study_name = f"hpo-study-{timestamp}"
        else:
            self.study_name = study_name

        self.storage = f"sqlite:///{storage_path}"
        self.study = None

    def _load_config(self, path):
        """تحميل ملف الإعدادات."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️ تحذير: ملف الإعدادات '{path}' غير موجود. سيتم استخدام القيم الافتراضية.")
            return {}

    def _get_sampler(self, sampler_name):
        """الحصول على كائن الـ sampler بناءً على الاسم."""
        sampler_name = sampler_name.lower()
        if sampler_name == "tpe":
            return optuna.samplers.TPESampler(**self.config.get('tpe_sampler', {}))
        elif sampler_name == "gp":
            return optuna.samplers.GPSampler(**self.config.get('gp_sampler', {}))
        elif sampler_name == "random":
            return optuna.samplers.RandomSampler()
        else:
            raise ValueError(f"Sampler '{sampler_name}' غير مدعوم. الخيارات المتاحة: TPE, GP, Random.")

    def _get_pruner(self, pruner_name):
        """الحصول على كائن الـ pruner بناءً على الاسم."""
        if pruner_name is None:
            return None
        pruner_name = pruner_name.lower()
        if pruner_name == "asha":
            return optuna.pruners.ASHApruner(**self.config.get('asha_pruner', {}))
        else:
            raise ValueError(f"Pruner '{pruner_name}' غير مدعوم. الخيار المتاح: ASHA.")

    def _objective_wrapper(self, trial):
        """
        غلاف لدالة الهدف لترجمة مساحة البحث.
        """
        params = {}
        for name, (param_type, low, high) in self.search_space.items():
            if param_type == "int":
                params[name] = trial.suggest_int(name, low, high)
            elif param_type == "float":
                params[name] = trial.suggest_float(name, low, high)
            elif param_type == "log":
                params[name] = trial.suggest_loguniform(name, low, high)
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, low)
            else:
                raise ValueError(f"نوع المعامل '{param_type}' غير مدعوم.")

        # هنا يتم استدعاء دالة الهدف الأصلية التي قدمها المستخدم
        # ولكن في هذا المثال، الأمثلة (sklearn/quick_start) تعرف suggest بأنفسها
        # لذا، سنقوم فقط بتمرير كائن التجربة مباشرة.
        return self.objective_function(trial)

    def optimize(self):
        """
        بدء عملية التحسين.
        """
        print(f"🚀 بدء الدراسة '{self.study_name}' مع {self.n_trials} تجربة.")
        print(f"   - الاتجاه: {self.direction}")
        print(f"   - Sampler: {self.sampler.__class__.__name__}")
        if self.pruner:
            print(f"   - Pruner: {self.pruner.__class__.__name__}")
        print(f"   - التوازي: {self.n_jobs} وظيفة")

        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True  # السماح باستئناف الدراسات
        )

        self.study.optimize(
            self._objective_wrapper,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )

        print("\n🎉 اكتملت عملية التحسين!")
        print(f"   - أفضل قيمة: {self.study.best_value}")
        print(f"   - أفضل المعاملات: {self.study.best_params}")

        self._generate_reports()

        return self.study.best_params, self.study.best_value

    def _generate_reports(self):
        """
        إنشاء تقارير نصية ورسوم بيانية.
        """
        report_dir = self.config.get("report_directory", "hpo_reports")
        os.makedirs(report_dir, exist_ok=True)

        study_report_dir = os.path.join(report_dir, self.study_name)
        os.makedirs(study_report_dir, exist_ok=True)

        print(f"\n📊 جارٍ إنشاء التقارير في المجلد: '{study_report_dir}'")

        # 1. تقرير نصي
        with open(os.path.join(study_report_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"ملخص دراسة التحسين: {self.study_name}\n")
            f.write("="*50 + "\n")
            f.write(f"الاتجاه: {self.direction}\n")
            f.write(f"عدد التجارب: {len(self.study.trials)}\n")
            f.write(f"أفضل قيمة: {self.study.best_value}\n\n")
            f.write("أفضل المعاملات:\n")
            for key, value in self.study.best_params.items():
                f.write(f"  - {key}: {value}\n")

            f.write("\nأفضل 10 تجارب:\n")
            sorted_trials = sorted(self.study.trials, key=lambda t: t.value, reverse=(self.direction=="maximize"))
            for i, trial in enumerate(sorted_trials[:10]):
                f.write(f"  {i+1}. القيمة: {trial.value}, المعاملات: {trial.params}\n")

        # 2. الرسوم البيانية
        try:
            # تاريخ التحسين
            fig_history = plot_optimization_history(self.study)
            fig_history.write_html(os.path.join(study_report_dir, "optimization_history.html"))

            # أهمية المعاملات
            if len(self.study.best_params) > 0:
                fig_importance = plot_param_importances(self.study)
                fig_importance.write_html(os.path.join(study_report_dir, "param_importances.html"))

            # شرائح المعاملات
            fig_slice = plot_slice(self.study)
            fig_slice.write_html(os.path.join(study_report_dir, "slice_plot.html"))

            print("   - ✅ تم إنشاء التقارير الرسومية بنجاح.")
        except Exception as e:
            print(f"   - ⚠️ فشل في إنشاء بعض الرسوم البيانية: {e}")

        print("   - ✅ تم إنشاء التقرير النصي بنجاح.")