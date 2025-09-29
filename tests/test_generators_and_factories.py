import unittest
import numpy as np

# Assuming sklearn is installed for these tests
try:
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from hpo.generators.parameter_generator import ParameterGenerator
from hpo.factories.model_factory import ModelFactory

class TestGeneratorsAndFactories(unittest.TestCase):

    def test_parameter_generator_suggests_sklearn_space(self):
        """Test if the generator creates a valid space for a sklearn RF."""
        pg = ParameterGenerator()
        dataset_meta = {'n_samples': 100, 'n_features': 10}
        space = pg.suggest_search_space('sklearn_rf_pipeline', dataset_meta)

        self.assertIn('n_estimators', space)
        self.assertEqual(space['n_estimators']['type'], 'int')
        self.assertEqual(len(space['n_estimators']['bounds']), 2)

        self.assertIn('max_features', space)
        self.assertEqual(space['max_features']['type'], 'categorical')
        self.assertListEqual(space['max_features']['choices'], ['sqrt', 'log2', None])

    def test_parameter_generator_suggests_keras_space(self):
        """Test if the generator creates a valid space for a Keras MLP."""
        pg = ParameterGenerator()
        dataset_meta = {'n_features': 32}
        space = pg.suggest_search_space('keras_mlp', dataset_meta)

        self.assertIn('hidden_layers', space)
        self.assertEqual(space['hidden_layers']['type'], 'int')

        self.assertIn('lr', space)
        self.assertEqual(space['lr']['type'], 'log_float')
        self.assertEqual(space['lr']['bounds'], (1e-5, 1e-1))

    def test_parameter_generator_sample(self):
        """Test the sampling method for different types."""
        pg = ParameterGenerator()
        self.assertIsInstance(pg.sample_from_spec({'type': 'int', 'bounds': (1, 10)}), int)
        self.assertIsInstance(pg.sample_from_spec({'type': 'float', 'bounds': (0.0, 1.0)}), float)
        self.assertIsInstance(pg.sample_from_spec({'type': 'categorical', 'choices': ['a', 'b']}), str)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
    def test_model_factory_builds_sklearn_pipeline(self):
        """Test if the factory can build a scikit-learn pipeline."""
        mf = ModelFactory(template_name='sklearn_rf_pipeline')
        params = {
            'n_estimators': 150,
            'max_depth': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt'
        }
        model = mf.build(params)
        self.assertIsInstance(model, Pipeline)
        self.assertIn('clf', model.named_steps)
        self.assertEqual(model.named_steps['clf'].n_estimators, 150)

    def test_model_factory_raises_for_unknown_template(self):
        """Test that the factory raises an error for an unknown template."""
        mf = ModelFactory(template_name='non_existent_model')
        with self.assertRaises(NotImplementedError):
            mf.build({})

    def test_model_factory_raises_for_unimplemented_template(self):
        """Test that the factory raises an error for a known but unimplemented template."""
        mf = ModelFactory(template_name='keras_mlp')
        with self.assertRaises(NotImplementedError):
            mf.build({})