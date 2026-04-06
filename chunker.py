def chunk_text(text, chunk_size=500, chunk_overlap=50, separators=None):
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    if len(text) <= chunk_size:
        return [text]

    sep = ""
    for s in separators:
        if s in text or s == "":
            sep = s
            break
    # Find the best separator, if text doesn't have \n\n, skip to next one

    if sep == "":
        pieces = list(text) # ["h", "e", "l", "l", "o"]
    else:
        pieces = text.split(sep)

    chunks = []
    current = pieces[0] 

    for piece in pieces[1:]:
        combine = current + sep + piece
        if len(combine) <= chunk_size:
            current = combine
        else:
            chunks.append(current)

            if chunk_overlap > 0 and len(current) > chunk_overlap:
                overlap_text = current[-chunk_overlap:]
                current = overlap_text + sep + piece
            else:
                current = piece

    chunks.append(current)

    remaining_seps = separators[separators.index(sep) + 1:] if sep in separators else separators[1:]
    final_chunks = []

    for chunk in chunks:
        if len(chunk) > chunk_size and remaining_seps:
            final_chunks.extend(chunk_text(chunk, chunk_size, chunk_overlap, remaining_seps))
        else:
            final_chunks.append(chunk)

    return final_chunks