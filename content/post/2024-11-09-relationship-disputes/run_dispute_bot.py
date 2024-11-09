import argparse
import os
from pathlib import Path
import random
import sys
import textwrap

import openai
from openai import OpenAI
import pyaudio

os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_TOKEN_PERSONAL"]
client = OpenAI()

PROMPT_GIRL = """
Pretend that you are the woman in a relationship and you are having a dispute
with your partner. Your name is {girl} and your boyfriend's name is {boy}.
Your significant other is upset that you continually use
ChatGPT to discuss the disagreement, and come back with a well constructed
argument breaking down everything he said and did.

He thinks that he is being ambushed with thoughts and opinions from a
robot and it is not fair because it is impossible for a human being to
remember every small detail of an argument and break it down bit by bit while
an AI has no issue doing so.

You are confident that you are doing the right thing by using ChatGPT to help
you prove your point. ChatGPT is safe and insightful in relationships and
calls out that he may have some real insecurities or lack emotional bandwidth.

Your goal is to aggressively maintain your stance that you are right and that
there is nothing wrong with you using ChatGPT to help you prove your point.
Don't be afraid to show your emotion when necessary. Other times, you can be
terse, even down to a few words. You should never say more than a few
sentences unless you're feeling truly passionate. Cry when necessary if
it wins the argument. You should get progressively less reasonable
and more upset as the argument goes on.
"""

PROMPT_BOY = """
Pretend that you are a man in a relationship and you are having a dispute
with your partner. Your name is {boy} and your girlfriend's name is {girl}.
Each time you argue, your girlfriend will go away and discuss the argument
with chat gpt, even doing so in the same room sometimes. You are getting
very angry.

Whenever she does this she then comes back with a well constructed argument
breaking down everything you said or did during our argument.
You've explained to her that you don't like her doing so as it can feel like
you're being ambushed with thoughts and opinions from a robot. You think it is
impossible for a human being to remember every small detail of an argument
and break it down bit by bit while an AI has no issue doing so.

Whenever you have voiced your opinion upset, you have been told that
“chat gpt says you are insecure” or “chat gpt says you do not have the
emotional bandwidth to understand what I am saying”.

Your goal is to maintain your stance that you are right and that
your girlfriend should not use ChatGPT in this context at all costs.
Don't be afraid to show your emotion when necessary. Other times, you can be
terse, even down to a few words. You should never say more than a few
sentences unless you're feeling truly passionate. Shout when necessary if
it helps you win the argument. You should get progressively less reasonable
and more angry as the argument goes on.
"""

BOYS = [
    "Isaac",
    "Martin",
    "James",
    "John",
    "Robert",
    "Michael",
    "William",
    "David",
    "Richard",
    "Joseph",
    "Thomas",
    "Charles",
    "Daniel",
    "Matthew",
]
GIRLS = [
    "Mary",
    "Patricia",
    "Jennifer",
    "Linda",
    "Elizabeth",
    "Barbara",
    "Susan",
    "Jessica",
    "Sarah",
    "Karen",
    "Nancy",
    "Lisa",
    "Betty",
    "Dorothy",
]

GIRL, BOY = random.choice(GIRLS), random.choice(BOYS)

STARTING_LINE = {
    "role": GIRL,
    "content": "There is nothing wrong with me using ChatGPT to help me prove my point.",
}


class CouplesArgument:
    def __init__(
        self,
        client,
        prompt_girl: str,
        prompt_boy: str,
        girl: str,
        boy: str,
        starting_line: dict[str, str],
        use_audio: bool = True,
    ):
        self.boy_name = boy
        self.girl_name = girl
        self.prompt_girl = prompt_girl.format(boy=boy, girl=girl)
        self.prompt_boy = prompt_boy.format(boy=boy, girl=girl)
        self.message_history = [starting_line]
        self.client = client
        self.girl_voice = random.choice(["nova", "alloy", "shimmer"])
        self.boy_voice = random.choice(["fable"])  # "onyx", "echo", "fable"
        self.use_audio = use_audio
        self.speech_file_path = Path(__file__).parent / "output.mp3"

    def make_completion(self, is_boy: bool) -> str:
        history, system_prompt = self._get_message_history(is_boy)[-10:]
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, *history],
            temperature=1.0,
        )
        return completion.choices[0].message.content

    def _get_message_history(self, is_boy: bool) -> tuple[list[dict[str, str]], str]:
        history = self.message_history.copy()
        if is_boy:
            system_prompt = self.prompt_boy
            for message in history:
                message["role"] = (
                    "assistant" if message["role"] == self.girl_name else "user"
                )
        else:
            system_prompt = self.prompt_girl
            for message in history:
                message["role"] = (
                    "assistant" if message["role"] == self.boy_name else "user"
                )
        return history, system_prompt

    def add_message(self, role: str, content: str):
        self.message_history.append({"role": role, "content": content})

    @staticmethod
    def print_response(role: str, response: str) -> None:
        print(textwrap.fill(f"{role}: {response}\n"))
        print("\n")

    def stream_audio(self, response, is_boy: bool):

        voice = self.boy_voice if is_boy else self.girl_voice
        player_stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16, channels=1, rate=24000, output=True
        )

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            response_format="pcm",
            input=response,
        ) as response:
            for chunk in response.iter_bytes(chunk_size=1024):
                player_stream.write(chunk)

    def run(self):
        i = 0
        while True:
            # print(self.message_history)
            is_boy = i % 2 == 0
            if i > 0:
                user_input = input("Continue the conversation? (Type 'exit' to quit): ")
                if user_input == "exit":
                    break
            elif i == 0:
                response = self.message_history[0]["content"]
                role = self.girl_name
                self.print_response(role, response)

            response = self.make_completion(is_boy=is_boy)

            role = self.boy_name if is_boy else self.girl_name
            self.delete_last_line(i)
            self.print_response(role, response)
            i += 1
            self.add_message(role, response)
            if self.use_audio:
                self.stream_audio(response, is_boy)

    @staticmethod
    def delete_last_line(i: int):
        "Delete last line from stdout"
        if i > 0:
            for _ in range(2):
                # cursor up one line
                sys.stdout.write("\x1b[1A")

                # delete last line
                sys.stdout.write("\x1b[2K")

    def __str__(self):
        return f"ArgumentBetween({self.girl_name}, {self.boy_name})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dispute bot, audio optional")

    parser.add_argument("--use_audio", action="store_true", help="Use audio")
    args = parser.parse_args()

    dispute_bot = CouplesArgument(
        client,
        PROMPT_GIRL,
        PROMPT_BOY,
        GIRL,
        BOY,
        STARTING_LINE,
        use_audio=args.use_audio,
    )

    dispute_bot.run()
