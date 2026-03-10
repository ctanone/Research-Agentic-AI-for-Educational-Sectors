import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI()

    async with client.realtime.connect(model="gpt-realtime") as connection:
        await connection.session.update(
            session={"type": "realtime", "output_modalities": ["text"]}
        )

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello!"}],
            }
        )
        await connection.response.create()

        async for event in connection:
            if event.type == "response.output_text.delta":
                print(event.delta, flush=True, end="")

            elif event.type == "response.output_text.done":
                print()

            elif event.type == "response.done":
                break

asyncio.run(main())