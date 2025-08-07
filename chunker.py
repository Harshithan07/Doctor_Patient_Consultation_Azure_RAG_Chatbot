import tiktoken

def split_text_into_chunks(text: str, max_tokens: int = 300) -> list[str]:
    """
    Split text into chunks based on token count.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of text chunks
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        candidate = current_chunk + "\n\n" + para
        num_tokens = len(encoding.encode(candidate))

        if num_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks