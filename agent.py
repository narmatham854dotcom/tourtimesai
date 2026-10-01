import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.services.sarvam.llm import SarvamLLMService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


async def bot(runner_args: RunnerArguments):
    """Main IRA Tour Times voice assistant."""

    transport = await create_transport(
        runner_args,
        {
            "exotel": lambda: FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
        },
    )

    stt = SarvamSTTService(
        api_key=os.getenv("SARVAM_API_KEY")
    )

    tts = SarvamTTSService(
        api_key=os.getenv("SARVAM_API_KEY")
    )

    llm = SarvamLLMService(
        api_key=os.getenv("SARVAM_API_KEY"),
        settings=SarvamLLMService.Settings(
            model="sarvam-105b"
        ),
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are IRA Tour Times, a friendly AI phone assistant "
                "for a tour and travel service. "
                "Keep responses short, clear, and conversational because "
                "the user is speaking with you over a phone call."
            ),
        }
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Caller connected")

        messages.append(
            {
                "role": "system",
                "content": (
                    "Greet the caller and briefly introduce yourself "
                    "as the IRA Tour Times assistant."
                ),
            }
        )

        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Caller disconnected")
        await task.cancel()

    runner = PipelineRunner(
        handle_sigint=runner_args.handle_sigint
    )

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
