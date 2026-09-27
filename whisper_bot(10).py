from __future__ import annotations

import os
import json
import logging
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, List

import discord
from discord import app_commands
from discord.ext import commands

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "whispers.json",
)

ANONYMOUS_NAME = "✦ Anonymous"
ANONYMOUS_COLOR = discord.Color.dark_grey()
ANONYMOUS_ICON: Optional[str] = None

GUILD_IDS: List[int] = [
    1543920625633071186
]

MAX_MESSAGE_LENGTH = 3800
SAVE_DEBOUNCE_SECONDS = 0.1
MESSAGE_CACHE_MAX_SIZE = 2000


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger("whisper-bot")


@dataclass
class WhisperRecord:
    message_id: int
    channel_id: int
    guild_id: int
    author_id: int
    original_content: str
    history: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    edited_at: Optional[str] = None

    @property
    def current_content(self) -> str:
        return self.history[-1] if self.history else self.original_content

    @property
    def is_edited(self) -> bool:
        return len(self.history) > 1

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "WhisperRecord":
        return WhisperRecord(**data)


class WhisperManager:
    def __init__(self, path: str):
        self.path = path

        self._records: Dict[str, WhisperRecord] = {}

        self._author_content_index: Dict[tuple, str] = {}
        self._channel_content_index: Dict[tuple, str] = {}

        self._message_cache: "OrderedDict[int, discord.Message]" = OrderedDict()

        self._lock = asyncio.Lock()

        self._save_task: Optional[asyncio.Task] = None
        self._save_generation = 0

        self.load()

    @staticmethod
    def _normalize_content(content: str) -> str:
        return content.strip()

    def _index_record(self, record: WhisperRecord) -> None:
        normalized = self._normalize_content(record.original_content)

        author_key = (record.author_id, record.channel_id, normalized)
        channel_key = (record.channel_id, normalized)

        self._author_content_index[author_key] = str(record.message_id)
        self._channel_content_index[channel_key] = str(record.message_id)

    def _rebuild_indexes(self) -> None:
        self._author_content_index.clear()
        self._channel_content_index.clear()

        records = sorted(self._records.values(), key=lambda r: r.created_at)

        for record in records:
            self._index_record(record)

    def load(self) -> None:
        if not os.path.exists(self.path):
            self._records = {}
            self._rebuild_indexes()
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if not isinstance(raw, dict):
                raise TypeError("whispers.json must contain an object")

            self._records = {
                str(k): WhisperRecord.from_dict(v) for k, v in raw.items()
            }

            self._rebuild_indexes()

            log.info("Loaded %d whisper(s) from %s", len(self._records), self.path)

        except (json.JSONDecodeError, OSError, TypeError, KeyError) as exc:
            log.warning(
                "Could not load %s (%s); starting with an empty store.",
                self.path,
                exc,
            )

            self._records = {}
            self._rebuild_indexes()

    def cache_message(self, message: discord.Message) -> None:
        self._message_cache[message.id] = message
        self._message_cache.move_to_end(message.id)

        while len(self._message_cache) > MESSAGE_CACHE_MAX_SIZE:
            self._message_cache.popitem(last=False)

    def get_cached_message(self, message_id: int) -> Optional[discord.Message]:
        message = self._message_cache.get(message_id)

        if message is not None:
            self._message_cache.move_to_end(message_id)

        return message

    def remove_cached_message(self, message_id: int) -> None:
        self._message_cache.pop(message_id, None)

    def _write_file(self, snapshot: dict) -> None:
        temp_path = f"{self.path}.tmp"

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, self.path)

        except OSError as exc:
            log.error("Could not save %s: %s", self.path, exc)

            try:
                os.remove(temp_path)
            except OSError:
                pass

    def _schedule_save(self) -> None:
        self._save_generation += 1

        if self._save_task is not None and not self._save_task.done():
            return

        self._save_task = asyncio.create_task(self._save_worker())
        self._save_task.add_done_callback(self._save_done_callback)

    def _save_done_callback(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Background whisper save failed.")

    async def _save_worker(self) -> None:
        while True:
            await asyncio.sleep(SAVE_DEBOUNCE_SECONDS)

            async with self._lock:
                generation = self._save_generation
                snapshot = {
                    key: record.to_dict() for key, record in self._records.items()
                }

            await asyncio.to_thread(self._write_file, snapshot)

            if generation == self._save_generation:
                return

    async def flush(self) -> None:
        task = self._save_task

        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def add(self, record: WhisperRecord) -> None:
        async with self._lock:
            key = str(record.message_id)
            self._records[key] = record
            self._index_record(record)

        self._schedule_save()

    async def remove(self, message_id: int) -> None:
        key = str(message_id)

        async with self._lock:
            record = self._records.pop(key, None)

            if record is not None:
                normalized = self._normalize_content(record.original_content)

                author_key = (record.author_id, record.channel_id, normalized)
                channel_key = (record.channel_id, normalized)

                if self._author_content_index.get(author_key) == key:
                    self._author_content_index.pop(author_key, None)

                if self._channel_content_index.get(channel_key) == key:
                    self._channel_content_index.pop(channel_key, None)

        self.remove_cached_message(message_id)
        self._schedule_save()

    async def update(self, record: WhisperRecord) -> None:
        async with self._lock:
            self._records[str(record.message_id)] = record
            self._index_record(record)

        self._schedule_save()

    def get(self, message_id: int) -> Optional[WhisperRecord]:
        return self._records.get(str(message_id))

    def find_by_original_content(
        self, *, author_id: int, channel_id: int, content: str
    ) -> Optional[WhisperRecord]:
        key = (author_id, channel_id, self._normalize_content(content))
        message_id = self._author_content_index.get(key)

        if message_id is None:
            return None

        return self._records.get(message_id)

    def find_any_by_original_content(
        self, *, channel_id: int, content: str
    ) -> Optional[WhisperRecord]:
        key = (channel_id, self._normalize_content(content))
        message_id = self._channel_content_index.get(key)

        if message_id is None:
            return None

        return self._records.get(message_id)

    def list_by_author(self, *, author_id: int, guild_id: int) -> List[WhisperRecord]:
        records = [
            record
            for record in self._records.values()
            if record.author_id == author_id and record.guild_id == guild_id
        ]

        records.sort(key=lambda r: r.created_at, reverse=True)
        return records


whisper_manager = WhisperManager(DATA_FILE)


intents = discord.Intents.default()
intents.message_content = True


class WhisperBot(commands.Bot):
    def __init__(self) -> None:
        super().__init__(command_prefix="!whisperbot-no-prefix!", intents=intents)

    async def close(self) -> None:
        await whisper_manager.flush()
        await super().close()

    async def setup_hook(self) -> None:
        if GUILD_IDS:
            for guild_id in GUILD_IDS:
                guild = discord.Object(id=guild_id)
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)

                log.info(
                    "Synced %d slash command(s) INSTANTLY on guild %s.",
                    len(synced),
                    guild_id,
                )

            self.tree.clear_commands(guild=None)
            await self.tree.sync()
        else:
            synced = await self.tree.sync()
            log.info("Synced %d slash command(s) GLOBALLY.", len(synced))


bot = WhisperBot()


@bot.event
async def on_ready() -> None:
    log.info("Logged in as %s (ID: %s)", bot.user, bot.user.id if bot.user else "?")

    try:
        await bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="/whisper",
            )
        )
    except discord.HTTPException:
        pass


def _author_kwargs() -> dict:
    kwargs = {"name": ANONYMOUS_NAME}

    if ANONYMOUS_ICON:
        kwargs["icon_url"] = ANONYMOUS_ICON

    return kwargs


def build_whisper_embed(content: str) -> discord.Embed:
    embed = discord.Embed(
        description=content,
        color=ANONYMOUS_COLOR,
        timestamp=datetime.now(timezone.utc),
    )

    embed.set_author(**_author_kwargs())
    embed.set_footer(text="WHISPER • ANONYMOUS MESSAGE")
    return embed


def build_history_embed(history: List[str]) -> discord.Embed:
    if len(history) <= 1:
        return build_whisper_embed(history[0] if history else "")

    parts = [history[0]]

    for idx, version in enumerate(history[1:], start=2):
        parts.append("━━━━━━━━━━━━━━━━━━")
        parts.append(f"✏️ **New version ({idx}):**\n{version}")

    description = "\n\n".join(parts)

    embed = discord.Embed(
        description=description,
        color=ANONYMOUS_COLOR,
        timestamp=datetime.now(timezone.utc),
    )

    embed.set_author(**_author_kwargs())
    embed.set_footer(text="WHISPER • ANONYMOUS MESSAGE • EDITED")
    return embed


def build_reply_embed(quoted_original: str, reply_text: str) -> discord.Embed:
    quoted_lines = "\n".join(
        f"> {line}" for line in quoted_original.splitlines()
    ) or "> "

    description = f"{quoted_lines}\n\n↳ **Reply:**\n{reply_text}"

    embed = discord.Embed(
        description=description,
        color=ANONYMOUS_COLOR,
        timestamp=datetime.now(timezone.utc),
    )

    embed.set_author(**_author_kwargs())
    embed.set_footer(text="WHISPER • ANONYMOUS REPLY")
    return embed


async def find_target_message(
    channel: discord.abc.Messageable, content: str, history_limit: int = 500
):
    target = content.strip()

    record = whisper_manager.find_any_by_original_content(
        channel_id=channel.id, content=target
    )

    if record is not None:
        cached = whisper_manager.get_cached_message(record.message_id)

        if cached is not None:
            return cached, record.current_content

        try:
            message = await channel.fetch_message(record.message_id)
            whisper_manager.cache_message(message)
            return message, record.current_content

        except discord.NotFound:
            await whisper_manager.remove(record.message_id)

        except discord.HTTPException:
            return None, None

    async for message in channel.history(limit=history_limit):
        if message.content.strip() == target:
            return message, message.content

        for embed in message.embeds:
            if embed.description and embed.description.strip() == target:
                return message, embed.description

    return None, None


@bot.tree.command(name="whisper", description="Send an anonymous message in this channel.")
@app_commands.describe(message="The message you want to send anonymously.")
async def whisper_command(interaction: discord.Interaction, message: str) -> None:
    if interaction.guild is None:
        await interaction.response.send_message(
            "This command only works in a server, not in direct messages.",
            ephemeral=True,
        )
        return

    channel = interaction.channel

    if not isinstance(channel, (discord.TextChannel, discord.Thread)):
        await interaction.response.send_message(
            "I can't send anonymous messages in this type of channel.",
            ephemeral=True,
        )
        return

    message = message.strip()

    if not message:
        await interaction.response.send_message("The message can't be empty.", ephemeral=True)
        return

    if len(message) > MAX_MESSAGE_LENGTH:
        await interaction.response.send_message(
            f"The message is too long (max {MAX_MESSAGE_LENGTH} characters).",
            ephemeral=True,
        )
        return

    embed = build_whisper_embed(message)

    try:
        sent_message = await channel.send(embed=embed)
        whisper_manager.cache_message(sent_message)

    except discord.Forbidden:
        await interaction.response.send_message(
            "I don't have permission to send messages in this channel.", ephemeral=True
        )
        return

    except discord.HTTPException as exc:
        log.error("Error sending whisper: %s", exc)
        await interaction.response.send_message("Something went wrong sending the message.", ephemeral=True)
        return

    record = WhisperRecord(
        message_id=sent_message.id,
        channel_id=channel.id,
        guild_id=interaction.guild.id,
        author_id=interaction.user.id,
        original_content=message,
        history=[message],
    )

    await whisper_manager.add(record)

    await interaction.response.send_message("✅ Your anonymous message was sent.", ephemeral=True)

    log.info(
        "Whisper sent by %s (%s) in #%s",
        interaction.user,
        interaction.user.id,
        getattr(channel, "name", channel.id),
    )


@bot.tree.command(
    name="delete",
    description="Delete an anonymous message you sent. Type the original message EXACTLY.",
)
@app_commands.describe(message="The EXACT original text of the anonymous message you want to delete.")
async def delete_command(interaction: discord.Interaction, message: str) -> None:
    if interaction.guild is None:
        await interaction.response.send_message("This command only works in a server.", ephemeral=True)
        return

    channel = interaction.channel

    record = whisper_manager.find_by_original_content(
        author_id=interaction.user.id,
        channel_id=channel.id,
        content=message,
    )

    if record is None:
        await interaction.response.send_message(
            "❌ I couldn't find any whisper of yours in this channel with that exact "
            "original text.\nMake sure you typed the message identically "
            "(capitalization, spaces, punctuation, emojis).",
            ephemeral=True,
        )
        return

    if not isinstance(channel, (discord.TextChannel, discord.Thread)):
        await interaction.response.send_message(
            "I can't delete messages in this type of channel.", ephemeral=True
        )
        return

    try:
        await channel.get_partial_message(record.message_id).delete()

    except discord.NotFound:
        pass

    except discord.Forbidden:
        await interaction.response.send_message("I don't have permission to delete that message.", ephemeral=True)
        return

    except discord.HTTPException as exc:
        log.error("Error deleting whisper %s: %s", record.message_id, exc)
        await interaction.response.send_message("Something went wrong deleting the message.", ephemeral=True)
        return

    await whisper_manager.remove(record.message_id)

    await interaction.response.send_message("🗑️ The anonymous message was deleted.", ephemeral=True)
    log.info("Whisper %s deleted by %s", record.message_id, interaction.user)


@bot.tree.command(name="edit", description="Edit an anonymous message you sent.")
@app_commands.describe(
    original_message="The EXACT original text of the message (as it was first sent).",
    new_message="The new text. Will appear BELOW the original, marked as 'New version'.",
)
async def edit_command(
    interaction: discord.Interaction, original_message: str, new_message: str
) -> None:
    if interaction.guild is None:
        await interaction.response.send_message("This command only works in a server.", ephemeral=True)
        return

    channel = interaction.channel

    record = whisper_manager.find_by_original_content(
        author_id=interaction.user.id,
        channel_id=channel.id,
        content=original_message,
    )

    if record is None:
        await interaction.response.send_message(
            "❌ I couldn't find any whisper of yours in this channel with that exact "
            "original text.\nMake sure you typed the ORIGINAL message identically.",
            ephemeral=True,
        )
        return

    new_message = new_message.strip()

    if not new_message:
        await interaction.response.send_message("The new message can't be empty.", ephemeral=True)
        return

    new_history = record.history + [new_message]
    new_embed = build_history_embed(new_history)

    if len(new_embed.description or "") > 4096:
        await interaction.response.send_message(
            "This message has accumulated too many edits and exceeded Discord's limit. "
            "Delete it with /delete and send a new one.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(ephemeral=True)

    discord_message = whisper_manager.get_cached_message(record.message_id)

    try:
        if discord_message is None:
            discord_message = await channel.fetch_message(record.message_id)
            whisper_manager.cache_message(discord_message)

        await discord_message.edit(embed=new_embed)

    except discord.NotFound:
        await interaction.followup.send("The original message no longer exists.", ephemeral=True)
        await whisper_manager.remove(record.message_id)
        return

    except discord.Forbidden:
        await interaction.followup.send("I don't have permission to edit that message.", ephemeral=True)
        return

    except discord.HTTPException as exc:
        log.error("Error editing whisper: %s", exc)
        await interaction.followup.send("Something went wrong editing the message.", ephemeral=True)
        return

    record.history = new_history
    record.edited_at = datetime.now(timezone.utc).isoformat()

    await whisper_manager.update(record)

    await interaction.followup.send(
        "✏️ The message was edited. The new version now appears below the original.",
        ephemeral=True,
    )


@bot.tree.command(name="my-whispers", description="Privately list your own anonymous whispers in this server.")
async def my_whispers_command(interaction: discord.Interaction) -> None:
    if interaction.guild is None:
        await interaction.response.send_message("This command only works in a server.", ephemeral=True)
        return

    records = whisper_manager.list_by_author(
        author_id=interaction.user.id, guild_id=interaction.guild.id
    )

    if not records:
        await interaction.response.send_message(
            "You haven't sent any whispers in this server yet.", ephemeral=True
        )
        return

    shown = records[:25]

    embed = discord.Embed(title="Your whispers", color=ANONYMOUS_COLOR)

    for record in shown:
        jump_url = (
            f"https://discord.com/channels/{record.guild_id}/"
            f"{record.channel_id}/{record.message_id}"
        )

        preview = record.original_content.strip()

        if len(preview) > 100:
            preview = preview[:97] + "..."

        label = "✏️ edited" if record.is_edited else "original"

        embed.add_field(
            name=f"#{record.channel_id} ({label})",
            value=f"{preview}\n[Jump to message]({jump_url})",
            inline=False,
        )

    if len(records) > len(shown):
        embed.set_footer(
            text=f"Showing your {len(shown)} most recent whispers out of {len(records)} total."
        )

    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(
    name="whisper-reply",
    description="Reply anonymously to an existing whisper in this channel.",
)
@app_commands.describe(
    target_text="Exact text of the message you want to reply to.",
    message="Your reply, posted anonymously under the original.",
)
async def whisper_reply_command(
    interaction: discord.Interaction, target_text: str, message: str
) -> None:
    if interaction.guild is None:
        await interaction.response.send_message("This command only works in a server.", ephemeral=True)
        return

    channel = interaction.channel

    if not isinstance(channel, (discord.TextChannel, discord.Thread)):
        await interaction.response.send_message(
            "I can't send anonymous messages in this type of channel.", ephemeral=True
        )
        return

    message = message.strip()

    if not message:
        await interaction.response.send_message("Your reply can't be empty.", ephemeral=True)
        return

    if len(message) > MAX_MESSAGE_LENGTH:
        await interaction.response.send_message(
            f"Your reply is too long (max {MAX_MESSAGE_LENGTH} characters).", ephemeral=True
        )
        return

    await interaction.response.defer(ephemeral=True)

    target_message, quoted_content = await find_target_message(channel, target_text)

    if target_message is None:
        await interaction.followup.send(
            "❌ I couldn't find any message in this channel with that exact text.",
            ephemeral=True,
        )
        return

    embed = build_reply_embed(quoted_content, message)

    try:
        sent_message = await channel.send(embed=embed, reference=target_message, mention_author=False)
        whisper_manager.cache_message(sent_message)

    except discord.Forbidden:
        await interaction.followup.send(
            "I don't have permission to send messages in this channel.", ephemeral=True
        )
        return

    except discord.HTTPException as exc:
        log.error("Error sending whisper reply: %s", exc)
        await interaction.followup.send("Something went wrong sending your reply.", ephemeral=True)
        return

    record = WhisperRecord(
        message_id=sent_message.id,
        channel_id=channel.id,
        guild_id=interaction.guild.id,
        author_id=interaction.user.id,
        original_content=message,
        history=[message],
    )

    await whisper_manager.add(record)

    await interaction.followup.send("✅ Your anonymous reply was sent.", ephemeral=True)


class WhisperReplyModal(discord.ui.Modal, title="Whisper Reply"):
    reply_text = discord.ui.TextInput(
        label="Your anonymous reply",
        style=discord.TextStyle.paragraph,
        max_length=MAX_MESSAGE_LENGTH,
        required=True,
    )

    def __init__(self, target_message: discord.Message, quoted_content: str) -> None:
        super().__init__()
        self.target_message = target_message
        self.quoted_content = quoted_content

    async def on_submit(self, interaction: discord.Interaction) -> None:
        message = str(self.reply_text.value).strip()

        if not message:
            await interaction.response.send_message("Your reply can't be empty.", ephemeral=True)
            return

        channel = self.target_message.channel
        embed = build_reply_embed(self.quoted_content, message)

        await interaction.response.defer(ephemeral=True)

        try:
            sent_message = await channel.send(
                embed=embed, reference=self.target_message, mention_author=False
            )
            whisper_manager.cache_message(sent_message)

        except discord.Forbidden:
            await interaction.followup.send(
                "I don't have permission to send messages in this channel.", ephemeral=True
            )
            return

        except discord.HTTPException:
            await interaction.followup.send("Something went wrong sending your reply.", ephemeral=True)
            return

        record = WhisperRecord(
            message_id=sent_message.id,
            channel_id=channel.id,
            guild_id=interaction.guild.id,
            author_id=interaction.user.id,
            original_content=message,
            history=[message],
        )

        await whisper_manager.add(record)

        await interaction.followup.send("✅ Your anonymous reply was sent.", ephemeral=True)


@bot.tree.context_menu(name="Whisper Reply")
async def whisper_reply_context_menu(interaction: discord.Interaction, message: discord.Message) -> None:
    if interaction.guild is None:
        await interaction.response.send_message("This command only works in a server.", ephemeral=True)
        return

    channel = message.channel

    if not isinstance(channel, (discord.TextChannel, discord.Thread)):
        await interaction.response.send_message(
            "I can't send anonymous messages in this type of channel.", ephemeral=True
        )
        return

    quoted_content = message.content

    if not quoted_content and message.embeds:
        quoted_content = message.embeds[0].description or ""

    if not quoted_content:
        quoted_content = "[no text content]"

    whisper_manager.cache_message(message)

    modal = WhisperReplyModal(target_message=message, quoted_content=quoted_content)
    await interaction.response.send_modal(modal)


class EditWhisperModal(discord.ui.Modal, title="Edit Whisper"):
    new_text = discord.ui.TextInput(
        label="New version",
        style=discord.TextStyle.paragraph,
        max_length=MAX_MESSAGE_LENGTH,
        required=True,
    )

    def __init__(self, target_message: discord.Message, record: WhisperRecord) -> None:
        super().__init__()
        self.target_message = target_message
        self.record = record
        self.new_text.default = record.current_content

    async def on_submit(self, interaction: discord.Interaction) -> None:
        new_text = str(self.new_text.value).strip()

        if not new_text:
            await interaction.response.send_message("The new message can't be empty.", ephemeral=True)
            return

        new_history = self.record.history + [new_text]
        new_embed = build_history_embed(new_history)

        if len(new_embed.description or "") > 4096:
            await interaction.response.send_message(
                "This message has accumulated too many edits and exceeded Discord's limit. "
                "Delete it and send a new one.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=True)

        try:
            await self.target_message.edit(embed=new_embed)

        except discord.NotFound:
            await interaction.followup.send("The original message no longer exists.", ephemeral=True)
            await whisper_manager.remove(self.record.message_id)
            return

        except discord.Forbidden:
            await interaction.followup.send("I don't have permission to edit that message.", ephemeral=True)
            return

        except discord.HTTPException:
            await interaction.followup.send("Something went wrong editing the message.", ephemeral=True)
            return

        self.record.history = new_history
        self.record.edited_at = datetime.now(timezone.utc).isoformat()

        await whisper_manager.update(self.record)

        await interaction.followup.send(
            "✏️ The message was edited. The new version now appears below the original.",
            ephemeral=True,
        )


@bot.tree.context_menu(name="Edit Whisper")
async def edit_whisper_context_menu(interaction: discord.Interaction, message: discord.Message) -> None:
    if interaction.guild is None:
        await interaction.response.send_message("This command only works in a server.", ephemeral=True)
        return

    record = whisper_manager.get(message.id)

    if record is None:
        await interaction.response.send_message(
            "This isn't a whisper message I'm tracking, so I can't edit it.", ephemeral=True
        )
        return

    if record.author_id != interaction.user.id:
        await interaction.response.send_message("You can only edit your own whispers.", ephemeral=True)
        return

    whisper_manager.cache_message(message)

    modal = EditWhisperModal(target_message=message, record=record)
    await interaction.response.send_modal(modal)


@bot.tree.context_menu(name="Delete Whisper")
async def delete_whisper_context_menu(interaction: discord.Interaction, message: discord.Message) -> None:
    if interaction.guild is None:
        await interaction.response.send_message("This command only works in a server.", ephemeral=True)
        return

    record = whisper_manager.get(message.id)

    if record is None:
        await interaction.response.send_message(
            "This isn't a whisper message I'm tracking, so I can't delete it.", ephemeral=True
        )
        return

    if record.author_id != interaction.user.id:
        await interaction.response.send_message("You can only delete your own whispers.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    try:
        await message.delete()

    except discord.NotFound:
        pass

    except discord.Forbidden:
        await interaction.followup.send("I don't have permission to delete that message.", ephemeral=True)
        return

    except discord.HTTPException as exc:
        log.error("Error deleting whisper from context menu: %s", exc)
        await interaction.followup.send("Something went wrong deleting the message.", ephemeral=True)
        return

    await whisper_manager.remove(record.message_id)

    await interaction.followup.send("🗑️ The anonymous message was deleted.", ephemeral=True)


@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction, error: app_commands.AppCommandError
) -> None:
    log.error("Slash command error: %s", error)

    try:
        if interaction.response.is_done():
            await interaction.followup.send(
                "Something went wrong running that command.", ephemeral=True
            )
        else:
            await interaction.response.send_message(
                "Something went wrong running that command.", ephemeral=True
            )
    except discord.HTTPException:
        pass


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit(
            "Missing bot token!\n"
            "Set the DISCORD_BOT_TOKEN environment variable, for example:\n"
            "  Windows (PowerShell): $env:DISCORD_BOT_TOKEN='your-token'\n"
            "  Linux / macOS: export DISCORD_BOT_TOKEN='your-token'\n"
            "Then run again: python whisper_bot.py"
        )

    bot.run(TOKEN)
