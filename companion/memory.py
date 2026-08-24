import time

from .constraint import band_archetype
from .embeddings import Embedder, cosine
from .methods import outcome_for
from .models import AffectState, Activation, Memory, RetrievedMemory

SALIENCE_DECAY_TABLE = [
    (0.9, 0.0),
    (0.7, 0.000333),
    (0.4, 0.00167),
    (0.1, 0.0067),
    (0.0, 0.5),
]
REHEARSAL_BOOST = 0.1
RECENCY_HALF_LIFE_HOURS = 72.0
W_SIMILARITY, W_SALIENCE, W_RECENCY, W_RESONANCE = 0.45, 0.25, 0.15, 0.15
MIN_SCORE = 0.15
MAX_RESULTS = 5
WORD_BUDGET = 600

BAND_TAGS = {
    "severe_negative": ["pain", "fear"],
    "mild_dislike": ["discomfort"],
    "neutral": [],
    "warm_positive": ["warmth"],
    "delight": ["joy"],
}
MOOD_VALENCE_NEG, MOOD_VALENCE_POS, MOOD_AROUSAL_HIGH = -0.3, 0.3, 0.6


def decay_rate_for(salience: float) -> float:
    for threshold, rate in SALIENCE_DECAY_TABLE:
        if salience >= threshold:
            return rate
    return 0.5


def write_salience(impact: float) -> float:
    return max(0.1, round(abs(impact), 2))


def effective_salience(m: Memory, now: float) -> float:
    age_days = max(0.0, (now - m.created_at) / 86400.0)
    boosted = m.salience * (1.0 + REHEARSAL_BOOST * m.access_count)
    return min(1.0, boosted * (1.0 - m.decay_rate) ** age_days)


def recency_score(m: Memory, now: float) -> float:
    age_hours = max(0.0, (now - m.last_accessed) / 3600.0)
    return 0.5 ** (age_hours / RECENCY_HALF_LIFE_HOURS)


def mood_tags(affect: AffectState) -> list[str]:
    tags: list[str] = []
    if affect.valence <= MOOD_VALENCE_NEG:
        tags.extend(["pain", "fear"])
    elif affect.valence >= MOOD_VALENCE_POS:
        tags.extend(["joy", "warmth"])
    if affect.arousal >= MOOD_AROUSAL_HIGH:
        tags.append("excitement")
    return tags


def emotional_tags_for(activation: Activation) -> list[str]:
    tags = list(BAND_TAGS.get(band_archetype(activation.impact), []))
    if activation.ambivalent:
        tags.append("conflict")
    return tags


def apply_rehearsal(m: Memory, now: float) -> None:
    m.access_count += 1
    m.last_accessed = now


def relative_time(created_at: float, now: float) -> str:
    days = int(max(0.0, now - created_at) // 86400)
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def _word_count(text: str) -> int:
    return len(text.split())


def retrieve(
    memories: list[Memory],
    query_embedding: list[float],
    affect: AffectState,
    now: float,
    embedder_name: str,
    suppress: set[str] | None = None,
) -> list[RetrievedMemory]:
    suppress = suppress or set()
    scored: list[tuple[float, Memory, dict[str, float]]] = []
    current_mood_tags = mood_tags(affect)

    for m in memories:
        if m.id in suppress:
            continue
        similarity = cosine(query_embedding, m.embedding) if m.embedder == embedder_name else 0.0
        salience = effective_salience(m, now)
        recency = recency_score(m, now)
        tag_intersection = len(set(current_mood_tags) & set(m.emotional_tags))
        resonance = tag_intersection / max(1, len(m.emotional_tags))
        score = W_SIMILARITY * similarity + W_SALIENCE * salience + W_RECENCY * recency + W_RESONANCE * resonance
        breakdown = {
            "similarity": round(similarity, 4),
            "salience": round(salience, 4),
            "recency": round(recency, 4),
            "resonance": round(resonance, 4),
        }
        scored.append((score, m, breakdown))

    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[RetrievedMemory] = []
    used_words = 0
    for score, m, breakdown in scored:
        if score < MIN_SCORE:
            break
        if len(results) >= MAX_RESULTS:
            break
        wc = _word_count(m.content)
        if used_words + wc > WORD_BUDGET:
            continue
        used_words += wc
        results.append(
            RetrievedMemory(
                memory_id=m.id,
                content=m.content,
                created_at=m.created_at,
                score=round(score, 4),
                breakdown=breakdown,
            )
        )

    return results


def episodic_memory(
    companion_id: str,
    user_input: str,
    activation: Activation,
    embedder: Embedder,
    now: float | None = None,
    method: str | None = None,
    session_id: str = "",
) -> Memory:
    now = now if now is not None else time.time()
    salience = write_salience(activation.impact)
    decay_rate = decay_rate_for(salience)
    emotional_tags = emotional_tags_for(activation)
    date_str = time.strftime("%Y-%m-%d", time.gmtime(now))
    if method:
        content = (
            f"On {date_str}, user said: {user_input!r}. "
            f"I reacted with {activation.archetype} ({outcome_for(activation.impact)})."
        )
    else:
        content = f"On {date_str}, user said: {user_input!r}. I reacted with {activation.archetype}."
    embedding = embedder.embed(content)
    return Memory(
        companion_id=companion_id,
        kind="episodic",
        content=content,
        embedding=embedding,
        embedder=embedder.name,
        salience=salience,
        decay_rate=decay_rate,
        emotional_tags=emotional_tags,
        created_at=now,
        last_accessed=now,
        session_id=session_id,
    )
