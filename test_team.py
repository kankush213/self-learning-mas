"""Check real graph execution with fake LLM responses and no API calls."""

import importlib
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage


app = importlib.import_module("self-learning-mas")


class TeamTests(unittest.TestCase):
    def test_agents_run_in_order_and_pass_their_work_forward(self):
        llm = Mock(spec=BaseChatModel)
        llm.invoke.side_effect = [
            AIMessage(content="Research notes about sunlight."),
            AIMessage(content="A draft about solar panels."),
            AIMessage(content="A clear final explanation."),
        ]
        state = {
            "topic": "Solar panels",
            "research": "",
            "draft": "",
            "final_answer": "",
        }

        with self.assertLogs(app.logger, level="INFO") as captured:
            team = app.build_team(llm)
            updates = list(team.stream(state, stream_mode="updates"))

        messages = [record.getMessage() for record in captured.records]
        self.assertEqual(messages, [
            "Building graph: START -> researcher -> writer -> reviewer -> END.",
            "Graph compiled and ready to run.",
            "[1/3] Researcher started. Reading state: topic.",
            "Researcher: calling OpenAI; waiting for research notes.",
            "Researcher finished. Returning state update: research (30 characters).",
            "[2/3] Writer started. Reading state: topic, research.",
            "Writer: calling OpenAI; waiting for a draft.",
            "Writer finished. Returning state update: draft (27 characters).",
            "[3/3] Reviewer started. Reading state: topic, research, draft.",
            "Reviewer: calling OpenAI; waiting for the final answer.",
            "Reviewer finished. Returning state update: final_answer (26 characters).",
        ])

        self.assertEqual(
            [next(iter(update)) for update in updates],
            ["researcher", "writer", "reviewer"],
        )
        self.assertEqual(llm.invoke.call_count, 3)
        prompts = [call.args[0][1][1] for call in llm.invoke.call_args_list]
        self.assertIn("Solar panels", prompts[0])
        self.assertIn("Research notes about sunlight.", prompts[1])
        self.assertIn("Research notes about sunlight.", prompts[2])
        self.assertIn("A draft about solar panels.", prompts[2])
        self.assertEqual(
            updates[-1]["reviewer"]["final_answer"],
            "A clear final explanation.",
        )

    def test_command_line_logs_completion_or_safe_api_failure(self):
        for fails in (False, True):
            with self.subTest(api_failure=fails):
                llm = Mock(spec=BaseChatModel)
                if fails:
                    llm.invoke.side_effect = app.OpenAIError("private-api-error-detail")
                else:
                    llm.invoke.side_effect = [
                        AIMessage(content="Notes"),
                        AIMessage(content="Draft"),
                        AIMessage(content="Answer"),
                    ]

                with (
                    patch.object(app, "load_dotenv"),
                    patch.dict(app.os.environ, {"OPENAI_API_KEY": "test-secret-key"}, clear=True),
                    patch.object(app, "ChatOpenAI", return_value=llm),
                    patch.object(app.sys, "argv", ["self-learning-mas.py"]),
                    redirect_stdout(StringIO()),
                    self.assertLogs(app.logger, level="INFO") as captured,
                ):
                    if fails:
                        with self.assertRaisesRegex(SystemExit, "OpenAI request failed"):
                            app.main()
                    else:
                        app.main()

                logs = "\n".join(captured.output)
                self.assertIn("Program started.", logs)
                self.assertNotIn("test-secret-key", logs)
                self.assertNotIn("private-api-error-detail", logs)
                if fails:
                    self.assertIn("ERROR", logs)
                    self.assertIn("Execution stopped", logs)
                    self.assertNotIn("END reached", logs)
                    self.assertNotIn("[2/3]", logs)
                else:
                    self.assertIn("State updated by researcher. Updated fields: research.", logs)
                    self.assertIn("END reached", logs)


if __name__ == "__main__":
    unittest.main()