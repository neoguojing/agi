import unittest
from unittest.mock import patch, MagicMock
import threading

# Assuming we import the TaskFactory and constants like TASK_LLM, TASK_EMBEDDING, etc.
from agi.tasks.task_factory import TaskFactory, TASK_LLM, TASK_EMBEDDING, TASK_LLM_WITH_HISTORY, TASK_LLM_WITH_RAG, TASK_TRANSLATE, TASK_IMAGE_GEN, TASK_TTS, TASK_SPEECH_TEXT, TASK_RETRIEVER, TASK_AGENT


class TestTaskFactory(unittest.TestCase):

    @patch('your_module.ChatOpenAI')  # Mock ChatOpenAI
    @patch('your_module.OllamaEmbeddings')  # Mock OllamaEmbeddings
    @patch('your_module.create_chat_with_history')  # Mock create_chat_with_history
    @patch('your_module.create_chat_with_rag')  # Mock create_chat_with_rag
    @patch('your_module.create_translate_chain')  # Mock create_translate_chain
    @patch('your_module.create_image_gen_chain')  # Mock create_image_gen_chain
    @patch('your_module.create_text2speech_chain')  # Mock create_text2speech_chain
    @patch('your_module.create_speech2text_chain')  # Mock create_speech2text_chain
    @patch('your_module.create_retriever')  # Mock create_retriever
    def test_create_task(self, mock_create_retriever, mock_create_speech2text_chain, mock_create_text2speech_chain, mock_create_image_gen_chain, mock_create_translate_chain, 
                         mock_create_chat_with_rag, mock_create_chat_with_history, mock_OllamaEmbeddings, mock_ChatOpenAI):
        
        # Mock the classes and return some dummy values
        mock_llm_instance = MagicMock()
        mock_embedding_instance = MagicMock()
        mock_create_chat_with_history.return_value = "chat_with_history_instance"
        mock_create_chat_with_rag.return_value = "chat_with_rag_instance"
        mock_create_translate_chain.return_value = "translate_chain_instance"
        mock_create_image_gen_chain.return_value = "image_gen_chain_instance"
        mock_create_text2speech_chain.return_value = "text2speech_chain_instance"
        mock_create_speech2text_chain.return_value = "speech2text_chain_instance"
        mock_create_retriever.return_value = "retriever_instance"
        mock_ChatOpenAI.return_value = mock_llm_instance
        mock_OllamaEmbeddings.return_value = mock_embedding_instance

        # Test for TASK_LLM
        llm_task = TaskFactory.create_task(TASK_LLM, model_name="custom_model")
        mock_ChatOpenAI.assert_called_once_with(model="custom_model", openai_api_key="OPENAI_API_KEY", base_url="OLLAMA_API_BASE_URL/v1/")
        self.assertEqual(llm_task, mock_llm_instance)

        # Test for TASK_EMBEDDING
        embedding_task = TaskFactory.create_task(TASK_EMBEDDING, model_name="custom_embedding_model")
        mock_OllamaEmbeddings.assert_called_once_with(model="custom_embedding_model", base_url="OLLAMA_API_BASE_URL")
        self.assertEqual(embedding_task, mock_embedding_instance)

        # Test for TASK_LLM_WITH_HISTORY
        history_task = TaskFactory.create_task(TASK_LLM_WITH_HISTORY)
        mock_create_chat_with_history.assert_called_once_with(mock_llm_instance)
        self.assertEqual(history_task, "chat_with_history_instance")

        # Test for TASK_LLM_WITH_RAG
        rag_task = TaskFactory.create_task(TASK_LLM_WITH_RAG, some_param="value")
        mock_create_chat_with_rag.assert_called_once_with(mock_llm_instance, mock_embedding_instance, {"some_param": "value"})
        self.assertEqual(rag_task, "chat_with_rag_instance")

        # Test for TASK_TRANSLATE
        translate_task = TaskFactory.create_task(TASK_TRANSLATE)
        mock_create_translate_chain.assert_called_once_with(mock_llm_instance)
        self.assertEqual(translate_task, "translate_chain_instance")

        # Test for TASK_IMAGE_GEN
        image_gen_task = TaskFactory.create_task(TASK_IMAGE_GEN)
        mock_create_image_gen_chain.assert_called_once_with(mock_llm_instance)
        self.assertEqual(image_gen_task, "image_gen_chain_instance")

        # Test for TASK_TTS
        tts_task = TaskFactory.create_task(TASK_TTS)
        mock_create_text2speech_chain.assert_called_once_with()
        self.assertEqual(tts_task, "text2speech_chain_instance")

        # Test for TASK_SPEECH_TEXT
        speech_text_task = TaskFactory.create_task(TASK_SPEECH_TEXT)
        mock_create_speech2text_chain.assert_called_once_with()
        self.assertEqual(speech_text_task, "speech2text_chain_instance")

        # Test for TASK_RETRIEVER
        retriever_task = TaskFactory.create_task(TASK_RETRIEVER, some_param="value")
        mock_create_retriever.assert_called_once_with(mock_llm_instance, mock_embedding_instance, kwargs={"some_param": "value"})
        self.assertEqual(retriever_task, "retriever_instance")

        # Test for TASK_AGENT (currently not implemented)
        agent_task = TaskFactory.create_task(TASK_AGENT)
        self.assertIsNone(agent_task)

    def test_create_task_cache(self):
        # Test that instances are cached correctly
        with patch('your_module.ChatOpenAI') as mock_ChatOpenAI, patch('your_module.OllamaEmbeddings') as mock_OllamaEmbeddings:
            mock_llm_instance = MagicMock()
            mock_embedding_instance = MagicMock()
            mock_ChatOpenAI.return_value = mock_llm_instance
            mock_OllamaEmbeddings.return_value = mock_embedding_instance

            # Create task once
            task1 = TaskFactory.create_task(TASK_LLM)
            task2 = TaskFactory.create_task(TASK_LLM)

            # Ensure the same instance is returned
            self.assertEqual(task1, task2)

            task3 = TaskFactory.create_task(TASK_EMBEDDING)
            task4 = TaskFactory.create_task(TASK_EMBEDDING)

            # Ensure the embedding task instance is also cached
            self.assertEqual(task3, task4)

    def test_create_task_lock(self):
        # Test that the lock mechanism works by simulating concurrent task creation
        with patch('your_module.ChatOpenAI') as mock_ChatOpenAI:
            mock_llm_instance = MagicMock()
            mock_ChatOpenAI.return_value = mock_llm_instance

            def create_task_concurrently():
                TaskFactory.create_task(TASK_LLM)

            # Simulate concurrent task creation with threads
            threads = [threading.Thread(target=create_task_concurrently) for _ in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # Ensure that only one instance is created, even with multiple threads
            mock_ChatOpenAI.assert_called_once()


if __name__ == '__main__':
    unittest.main()
