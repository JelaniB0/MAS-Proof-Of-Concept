from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentbalance.cli import main
from agentbalance.env import load_env
from agentbalance.providers import connection
from agentbalance.config import Model


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / ".env"
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def write(self, source):
        self.path.write_text(source, encoding="utf-8-sig")

    def test_quotes_comments_export_and_literal_values(self):
        self.write('''# comment
export GROQ_API_KEY = "key=#part $literal" # note
AZURE_API_KEY='azure-secret'
MODEL = model-name # note
LITERAL=$(do-not-run)\\path
EMPTY=
COMMENT= # empty
''')
        load_env(self.path)
        self.assertEqual(os.environ["GROQ_API_KEY"], "key=#part $literal")
        self.assertEqual(os.environ["AZURE_API_KEY"], "azure-secret")
        self.assertEqual(os.environ["MODEL"], "model-name")
        self.assertEqual(os.environ["LITERAL"], "$(do-not-run)\\path")
        self.assertEqual(os.environ["EMPTY"], "")
        self.assertEqual(os.environ["COMMENT"], "")

    def test_existing_environment_takes_precedence(self):
        self.write("GROQ_API_KEY=file-key\nEMPTY=file-value\n")
        os.environ.update(GROQ_API_KEY="shell-key", EMPTY="")
        load_env(self.path)
        self.assertEqual(os.environ["GROQ_API_KEY"], "shell-key")
        self.assertEqual(os.environ["EMPTY"], "")

    def test_missing_default_is_optional_but_explicit_path_is_required(self):
        load_env(self.path)
        with self.assertRaisesRegex(ValueError, "Environment file not found"):
            load_env(self.path, required=True)

    def test_invalid_file_is_atomic_and_error_does_not_echo_secrets(self):
        for invalid in ('BROKEN secret-do-not-log', 'KEY="secret-do-not-log', 'BAD-KEY=secret-do-not-log',
                        'KEY=secret-do-not-log\x00'):
            with self.subTest(invalid=invalid):
                self.write("GOOD=first\n" + invalid)
                with self.assertRaises(ValueError) as error:
                    load_env(self.path)
                self.assertNotIn("secret-do-not-log", str(error.exception))
                self.assertNotIn("GOOD", os.environ)

    def test_duplicate_keys_use_last_file_value(self):
        self.write("MODEL=first\nMODEL=second\n")
        load_env(self.path)
        self.assertEqual(os.environ["MODEL"], "second")

    def test_cli_automatically_loads_working_directory_env(self):
        self.write("GROQ_API_KEY=from-default-file\n")
        with patch("agentbalance.cli.Path.cwd", return_value=self.path.parent), redirect_stdout(StringIO()):
            self.assertEqual(main(["plan", "--task", "test"]), 0)
        self.assertEqual(os.environ["GROQ_API_KEY"], "from-default-file")

    def test_cli_custom_file_replaces_default(self):
        self.write("GROQ_API_KEY=default\n")
        custom = self.path.with_name(".env.azure")
        custom.write_text("AZURE_API_KEY=custom\n", encoding="utf-8")
        with patch("agentbalance.cli.Path.cwd", return_value=self.path.parent), redirect_stdout(StringIO()):
            self.assertEqual(main(["plan", "--env-file", str(custom), "--task", "test"]), 0)
        self.assertEqual(os.environ["AZURE_API_KEY"], "custom")
        self.assertNotIn("GROQ_API_KEY", os.environ)

    def test_cli_no_env_skips_loading(self):
        with patch("agentbalance.cli.load_env") as loader, redirect_stdout(StringIO()):
            self.assertEqual(main(["plan", "--no-env", "--task", "test"]), 0)
        loader.assert_not_called()

    def test_cli_missing_explicit_file_fails_before_run(self):
        with patch("agentbalance.cli.execute") as execute, redirect_stderr(StringIO()):
            self.assertEqual(main(["run", "--env-file", str(self.path), "--task", "test"]), 2)
        execute.assert_not_called()

    def test_credentials_and_model_reach_provider_without_network(self):
        self.write("GROQ_API_KEY=fixture-key\nGROQ_CHEAP_MODEL=fixture-model\n")
        load_env(self.path)
        model = Model("cheap", "groq", "${GROQ_CHEAP_MODEL}", .4, 1, 2, 1,
                      endpoint="https://api.groq.com/openai/v1/chat/completions", api_key_env="GROQ_API_KEY")
        _, name, headers = connection(model)
        self.assertEqual(name, "fixture-model")
        self.assertEqual(headers["Authorization"], "Bearer fixture-key")


if __name__ == "__main__":
    unittest.main()
