from enum import Enum

DEFAULT_OPENAI_SEED_VALUE = 10

class GPT_Model(Enum):
    GPT_40_MINI = "gpt-4o-mini"
    GPT_03_MINI = "o3-mini"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_40 = "gpt-4o"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_NANO = "gpt-4.1-nano"

    @classmethod
    def values(cls):
        return [i.value for i in cls]

class Claude_Model(Enum):
    CLAUDE_3_5_HAIKU = "claude-3.5-haiku"
    CLAUDE_3_5_SONNET = "claude-3.5-sonnet"

    @classmethod
    def values(cls):
        return [i.value for i in cls]

class Google_LLM_Model(Enum):
    GEMINI_FLASH = "gemini-2.5-flash"
    GEMINI_PRO = "gemini-2.5-pro"

    @classmethod
    def values(cls):
        return [i.value for i in cls]

# These costs are in USD for per 1 million tokens
# Model costs were last updated on July 14, 2025
MODEL_COSTS = {
    # GPT Models
    GPT_Model.GPT_40_MINI.value: {
        "input_cost_per_1m_token": 0.60,
        "output_cost_per_1m_token": 2.40,
    },
    GPT_Model.GPT_03_MINI.value: {
        "input_cost_per_1m_token": 1.10,
        "output_cost_per_1m_token": 4.40,
    },
    GPT_Model.GPT_3_5_TURBO.value: {
        "input_cost_per_1m_token": 0.50,
        "output_cost_per_1m_token": 1.50,
    },
    GPT_Model.GPT_40.value: {
        "input_cost_per_1m_token": 2.50,
        "output_cost_per_1m_token": 10.00,
    },
    GPT_Model.GPT_4_1.value: {
        "input_cost_per_1m_token": 2.00,
        "output_cost_per_1m_token": 8.00,
    },
    GPT_Model.GPT_4_1_NANO.value: {
        "input_cost_per_1m_token": 0.10,
        "output_cost_per_1m_token": 0.40,
    },
    # Claude Models
    Claude_Model.CLAUDE_3_5_HAIKU.value: {
        "input_cost_per_1m_token": 0.80,
        "output_cost_per_1m_token": 4.00,
    },
    Claude_Model.CLAUDE_3_5_SONNET.value: {
        "input_cost_per_1m_token": 3.00,
        "output_cost_per_1m_token": 15.00,
    },
    # Google LLM Models
    Google_LLM_Model.GEMINI_FLASH.value: {
        "input_cost_per_1m_token": 0.30,
        "output_cost_per_1m_token": 2.50,
    },
    Google_LLM_Model.GEMINI_PRO.value: {
        "input_cost_per_1m_token": 2.50,
        "output_cost_per_1m_token": 10.00,
    },
}


ONE_MILLION = 1000000

# Exchange rate last updated on July 14, 2025
USD_TO_INR_EXCHANGE_RATE = 85.98

# Default OpenAI Model in use
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Usage
# cost_inr = MODEL_COSTS["gpt-3.5-turbo"]["input_cost_per_1m_token"] * USD_TO_INR_EXCHANGE_RATE
