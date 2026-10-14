# Integration receipts: group-chat-mcp as read for CAT-4B

Read-only reading of `~/group-chat-mcp` at commit `b54ff5b` (2026-10-08), copied verbatim. group-chat-mcp is never
modified for this project; these lines are what the served model has to fit.

## group-chat tool loop reading

`services/ai_gateway.py:296-298`, the multimodal check is a substring match on the model name:

```python
    def _is_multimodal(model: str) -> bool:
        m = model.lower()
        return any(k in m for k in ("gemini", "claude", "grok-4", "gpt-4o", "pixtral", "-vl", "llava"))
```

`services/ai_gateway.py`, `_generate_with_tools`: a round that makes tool calls is appended to the model's history
(`messages.append(choice.message.model_dump())`) and the loop continues; only a round without tool calls returns, and
it returns that round's text:

```python
                if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                    content = choice.message.content or ""
                    if content:
                        return content
```

So text the model writes beside a tool call (round 1) never reaches the chat; the chat sees the final reply.

`services/ai_gateway.py:427-434`, after each tool round a multimodal model gets every freshly generated image back as
a **user** turn:

```python
                if self._is_multimodal(model):
                    for path in imagegen_service.peek_pending()[imgs_before:]:
                        ...
                            messages.append({"role": "user", "content": [
                                {"type": "text", "text": "(Image you just generated, for your review — iterate if it's not right:)"},
```

`services/context_builder.py:86-88`, an image-intent turn carries the chat's latest three images:

```python
            packet.image_intent = _image_relevant(msg.text) or getattr(msg, "has_images", False)
            if packet.image_intent:
                packet.input_images = latest_image_locals(msg.chat_id, 3)
```

With up to three review images in a round, `serve/vllm.sh` allows 6 images per prompt.
