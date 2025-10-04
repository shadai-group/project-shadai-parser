import os
import time

from load_dotenv import load_dotenv

from parser_shadai import AgentConfig, GeminiProvider, MainProcessingAgent

load_dotenv()

init = time.time()
print("Starting...")
config = AgentConfig(
    chunk_size=1000,
    overlap_size=200,
    temperature=0.2,
    extract_images=False,
    auto_detect_language=True,
    extract_text_from_images=False,
)

api_key = os.getenv("GOOGLE_API_KEY")

agent = MainProcessingAgent(llm_provider=GeminiProvider(api_key=api_key), config=config)

result = agent.process_file("SPA-Constiution.pdf")
print(result.get("chunks")[0].get("metadata").get("summary"))
print(result.get("chunks")[-1].get("metadata").get("summary"))
print(f"Time taken: {time.time() - init} seconds")
print("Done")
