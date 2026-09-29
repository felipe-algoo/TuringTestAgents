import sys
import hashlib
import time
import random
import json
import pickle
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, jensenshannon
from scipy.stats import entropy, ks_2samp

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QComboBox, QSlider, QFileDialog, QStatusBar,
    QMessageBox, QCheckBox, QTabWidget, QScrollArea, QFrame,
    QSizePolicy, QToolTip
)
from PySide6.QtCore import Qt, Signal, QThread, QObject, QSize
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QKeySequence, QTextCursor, QShortcut

@dataclass
class Message:
    sender: str
    text: str
    timestamp: float
    features: Dict[str, float] = field(default_factory=dict)

@dataclass
class Episode:
    state_hash: str
    embedding: np.ndarray
    analysis_result: Dict[str, Any]
    context_summary: str
    timestamp: float
    conversation_length: int

@dataclass
class MetricPoint:
    turn: int
    composite: float
    stylometric: float
    entropy_diff: float
    ks_score: float
    js_div: float
    ngram: float
    regularity: float

class DeterministicEmbedder:
    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "big") % (2 ** 32)
        rng = np.random.RandomState(seed)
        vec = rng.normal(0.0, 1.0, self.dim)
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        return vec

    def state_hash(self, state: Dict[str, Any]) -> str:
        payload = str(sorted((k, str(v)) for k, v in state.items())).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

class EpisodicSemanticMemory:
    def __init__(self, capacity: int = 200, dim: int = 64):
        self.capacity = max(10, capacity)
        self.embedder = DeterministicEmbedder(dim)
        self.episodes: List[Episode] = []
        self.df = pd.DataFrame(columns=[
            "state_hash", "result_label", "confidence",
            "context_summary", "timestamp", "conversation_length"
        ])

    def set_capacity(self, capacity: int):
        self.capacity = max(10, capacity)
        if len(self.episodes) > self.capacity:
            self.episodes = self.episodes[-self.capacity:]
        if len(self.df) > self.capacity:
            self.df = self.df.iloc[-self.capacity:].reset_index(drop=True)

    def store(self, state: Dict[str, Any], result: Dict[str, Any], context: str, conv_len: int):
        h = self.embedder.state_hash(state)
        emb = self.embedder.embed(h + context)
        ep = Episode(
            state_hash=h,
            embedding=emb,
            analysis_result=result,
            context_summary=context,
            timestamp=time.time(),
            conversation_length=conv_len
        )
        self.episodes.append(ep)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
        row = {
            "state_hash": h,
            "result_label": str(result.get("label", "unknown")),
            "confidence": float(result.get("confidence", 0.0)),
            "context_summary": str(context)[:200],
            "timestamp": float(ep.timestamp),
            "conversation_length": int(conv_len)
        }
        new_df = pd.DataFrame([row])
        if self.df.empty:
            self.df = new_df
        else:
            self.df = pd.concat([self.df, new_df], ignore_index=True)
        if len(self.df) > self.capacity:
            self.df = self.df.iloc[-self.capacity:].reset_index(drop=True)

    def retrieve(self, state: Dict[str, Any], top_k: int = 5, decay: float = 0.15) -> List[Episode]:
        if not self.episodes:
            return []
        h = self.embedder.state_hash(state)
        query = self.embedder.embed(h)
        now = time.time()
        scored = []
        for ep in self.episodes:
            sim = 1.0 - cosine(query, ep.embedding)
            age_hours = max(0.0, (now - ep.timestamp) / 3600.0)
            weight = np.exp(-decay * age_hours)
            scored.append((sim * weight, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        k = max(1, min(top_k, len(scored)))
        return [ep for _, ep in scored[:k]]

    def inject_context(self, retrieved: List[Episode], lang: str = "en") -> str:
        if not retrieved:
            return "Nenhum contexto episódico anterior disponível." if lang == "pt" else "No prior episodic context available."
        parts = []
        for i, ep in enumerate(retrieved):
            label = ep.analysis_result.get("label", "?")
            conf = ep.analysis_result.get("confidence", 0.0)
            parts.append(f"E{i+1}:{label}@{conf:.2f} L{ep.conversation_length}")
        return " | ".join(parts)

    def clear(self):
        self.episodes.clear()
        self.df = pd.DataFrame(columns=[
            "state_hash", "result_label", "confidence",
            "context_summary", "timestamp", "conversation_length"
        ])

    def save(self, path: str):
        data = {
            "capacity": self.capacity,
            "episodes": [
                {
                    "state_hash": ep.state_hash,
                    "embedding": ep.embedding.tolist(),
                    "analysis_result": ep.analysis_result,
                    "context_summary": ep.context_summary,
                    "timestamp": ep.timestamp,
                    "conversation_length": ep.conversation_length
                }
                for ep in self.episodes
            ]
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.capacity = data.get("capacity", 200)
        self.episodes = []
        for item in data.get("episodes", []):
            ep = Episode(
                state_hash=item["state_hash"],
                embedding=np.array(item["embedding"], dtype=float),
                analysis_result=item["analysis_result"],
                context_summary=item["context_summary"],
                timestamp=item["timestamp"],
                conversation_length=item["conversation_length"]
            )
            self.episodes.append(ep)
        rows = []
        for ep in self.episodes:
            rows.append({
                "state_hash": ep.state_hash,
                "result_label": str(ep.analysis_result.get("label", "unknown")),
                "confidence": float(ep.analysis_result.get("confidence", 0.0)),
                "context_summary": ep.context_summary[:200],
                "timestamp": float(ep.timestamp),
                "conversation_length": int(ep.conversation_length)
            })
        self.df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            "state_hash", "result_label", "confidence",
            "context_summary", "timestamp", "conversation_length"
        ])

    def stats(self) -> Dict[str, float]:
        if not self.episodes:
            return {"count": 0, "mean_conf": 0.0, "distinct_rate": 0.0, "mean_len": 0.0}
        confs = [ep.analysis_result.get("confidence", 0.0) for ep in self.episodes]
        labels = [ep.analysis_result.get("label", "") for ep in self.episodes]
        lengths = [ep.conversation_length for ep in self.episodes]
        distinct = sum(1 for l in labels if l == "distinct")
        return {
            "count": float(len(self.episodes)),
            "mean_conf": float(np.mean(confs)),
            "distinct_rate": distinct / max(len(labels), 1),
            "mean_len": float(np.mean(lengths))
        }

class ProbabilisticAgent:
    MODELS = ("analytical", "intuitive", "concise", "verbose")

    def __init__(self, name: str, model_type: str = "analytical", seed: int = 42, lang: str = "en"):
        self.name = name
        self.model_type = model_type if model_type in self.MODELS else "analytical"
        self.lang = lang
        self.seed = seed % (2 ** 32)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.RandomState(self.seed)
        self.history: List[Message] = []
        self.state = {
            "formality": 0.5,
            "verbosity": 0.5,
            "agreement": 0.5,
            "curiosity": 0.5,
            "emotionality": 0.5
        }
        self._init_model()

    def set_language(self, lang: str):
        self.lang = lang
        self._init_model()

    def set_model(self, model_type: str):
        if model_type in self.MODELS:
            self.model_type = model_type
            self._init_model()

    def set_params(self, formality: float, verbosity: float, agreement: float, curiosity: float, emotionality: float):
        self.state["formality"] = float(np.clip(formality, 0.0, 1.0))
        self.state["verbosity"] = float(np.clip(verbosity, 0.0, 1.0))
        self.state["agreement"] = float(np.clip(agreement, 0.0, 1.0))
        self.state["curiosity"] = float(np.clip(curiosity, 0.0, 1.0))
        self.state["emotionality"] = float(np.clip(emotionality, 0.0, 1.0))

    def set_seed(self, seed: int):
        self.seed = seed % (2 ** 32)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.RandomState(self.seed)

    def _init_model(self):
        pt = self.lang == "pt"
        if self.model_type == "analytical":
            self.state.update({"formality": 0.85, "verbosity": 0.65, "agreement": 0.35, "curiosity": 0.8, "emotionality": 0.15})
            if pt:
                self.templates = {
                    "greeting": ["Saudações. Iniciamos uma troca estruturada?", "Olá. Examinemos o tema metodicamente.", "Bom dia. Qual o objeto preciso de análise?"],
                    "response": ["Com base nos dados, {point}.", "Analiticamente, {point}.", "As evidências indicam {point}.", "Logicamente, {point}.", "Dados os parâmetros, {point}."],
                    "question": ["Qual sua avaliação quantitativa de {topic}?", "Elabore a estrutura causal de {topic}.", "Como interagem as variáveis em {topic}?"],
                    "agreement": ["Alinha-se ao padrão observado.", "Raciocínio consistente.", "Concordo com a análise formal."],
                    "disagreement": ["Os dados não suportam plenamente.", "Há inconsistência lógica.", "Hipóteses alternativas permanecem."]
                }
                self.points = ["a distribuição não é uniforme", "correlação não implica causalidade", "hipótese nula não rejeitada", "variância acima do esperado", "modelo requer restrições", "entropia crescente", "amostra insuficiente", "possível multicolinearidade"]
                self.topics = ["significância estatística", "complexidade do modelo", "interação de features", "ganho de informação", "fronteiras de decisão"]
            else:
                self.templates = {
                    "greeting": ["Greetings. Shall we begin a structured exchange?", "Hello. I propose methodical examination.", "Good day. What is the precise subject?"],
                    "response": ["Based on available data, {point}.", "Analytically, {point}.", "Evidence suggests {point}.", "Logically, {point}.", "Given the parameters, {point}."],
                    "question": ["What is your quantitative assessment of {topic}?", "Elaborate the causal structure of {topic}.", "How do variables interact on {topic}?"],
                    "agreement": ["Aligns with the observed pattern.", "Reasoning is consistent.", "I concur with the formal analysis."],
                    "disagreement": ["Data does not fully support that.", "Logical inconsistency appears.", "Alternative hypotheses remain viable."]
                }
                self.points = ["the distribution is non-uniform", "correlation does not imply causation", "null hypothesis cannot be rejected", "variance exceeds bounds", "model needs constraints", "system entropy is rising", "sample size insufficient", "multicollinearity may exist"]
                self.topics = ["statistical significance", "model complexity", "feature interaction", "information gain", "decision boundaries"]
        elif self.model_type == "intuitive":
            self.state.update({"formality": 0.25, "verbosity": 0.45, "agreement": 0.7, "curiosity": 0.65, "emotionality": 0.8})
            if pt:
                self.templates = {
                    "greeting": ["E aí! Pronto pra conversar?", "Oi! No que está pensando?", "Olá, vamos falar!"],
                    "response": ["Sinto que {point}.", "Isso me lembra {point}.", "{point} tem outro impacto.", "Sinceramente, {point}.", "Meio que acho {point}."],
                    "question": ["O que sente sobre {topic}?", "{topic} faz sentido pra você?", "Como {topic} te afeta?"],
                    "agreement": ["Entendo totalmente!", "Mesma vibe, parece certo.", "Estou contigo."],
                    "disagreement": ["Não sei se concordo.", "Parece estranho.", "Algo não encaixa."]
                }
                self.points = ["as coisas estão mais conectadas", "intuição supera lógica pura", "histórias carregam mais verdade", "pessoas mudam com o contexto", "momentos quietos importam", "energia segue a atenção", "padrões se escondem", "equilíbrio nunca é estático"]
                self.topics = ["sentimentos", "experiências", "padrões ocultos", "ressonância", "observações"]
            else:
                self.templates = {
                    "greeting": ["Hey there! Ready to chat?", "Hi! What's on your mind?", "Hello friend, let's talk!"],
                    "response": ["I feel like {point}.", "That reminds me of {point}.", "{point} just hits different.", "Honestly, {point}.", "I kinda think {point}."],
                    "question": ["What do you feel about {topic}?", "Does {topic} make sense to you?", "How does {topic} sit with you?"],
                    "agreement": ["Yeah, I totally get that!", "Same here, feels right.", "Absolutely with you."],
                    "disagreement": ["Not sure I vibe with that.", "Feels off somehow.", "Something doesn't click."]
                }
                self.points = ["things are more connected than they seem", "intuition often beats pure logic", "stories carry more truth than numbers", "people change with context", "quiet moments matter most", "energy flows where attention goes", "patterns hide in plain sight", "balance is never static"]
                self.topics = ["gut feelings", "shared experiences", "hidden patterns", "emotional resonance", "quiet observations"]
        elif self.model_type == "concise":
            self.state.update({"formality": 0.6, "verbosity": 0.2, "agreement": 0.5, "curiosity": 0.4, "emotionality": 0.2})
            if pt:
                self.templates = {
                    "greeting": ["Olá.", "Iniciemos.", "Pronto."],
                    "response": ["{point}.", "Fato: {point}.", "Resumo: {point}."],
                    "question": ["E {topic}?", "Detalhe {topic}.", "{topic}?"],
                    "agreement": ["De acordo.", "Ok.", "Correto."],
                    "disagreement": ["Discordo.", "Não.", "Impreciso."]
                }
                self.points = ["dados insuficientes", "padrão claro", "ruído alto", "sinal fraco", "conclusão prematura", "amostra enviesada", "variável oculta", "limite atingido"]
                self.topics = ["resultado", "método", "erro", "limite", "amostra"]
            else:
                self.templates = {
                    "greeting": ["Hello.", "Begin.", "Ready."],
                    "response": ["{point}.", "Fact: {point}.", "Summary: {point}."],
                    "question": ["And {topic}?", "Detail {topic}.", "{topic}?"],
                    "agreement": ["Agreed.", "OK.", "Correct."],
                    "disagreement": ["Disagree.", "No.", "Imprecise."]
                }
                self.points = ["insufficient data", "clear pattern", "high noise", "weak signal", "premature conclusion", "biased sample", "hidden variable", "limit reached"]
                self.topics = ["result", "method", "error", "bound", "sample"]
        else:
            self.state.update({"formality": 0.4, "verbosity": 0.9, "agreement": 0.55, "curiosity": 0.85, "emotionality": 0.5})
            if pt:
                self.templates = {
                    "greeting": ["Olá! Que bom conversar longamente sobre ideias complexas.", "Saudações! Vamos explorar cada nuance do tema.", "Bom dia! Há muito a dissecar juntos."],
                    "response": ["Considerando múltiplas camadas, {point}, e isso se conecta a outras observações relevantes.", "Em detalhe, {point}, o que abre novas perguntas sobre o sistema.", "Expandindo: {point}, além de implicações secundárias importantes."],
                    "question": ["Poderia desenvolver extensamente sua visão sobre {topic} e suas ramificações?", "Quais aspectos de {topic} ainda não exploramos adequadamente?", "Como {topic} interage com fatores contextuais mais amplos?"],
                    "agreement": ["Concordo e gostaria de adicionar camadas à sua análise.", "Sim, e podemos aprofundar ainda mais esse ponto.", "Alinhado, com ressalvas interessantes a considerar."],
                    "disagreement": ["Discordo em parte, pois há dimensões adicionais a examinar.", "Não completamente, vejamos contraexemplos e matizes.", "Há tensão com outras evidências que merecem discussão."]
                }
                self.points = ["o sistema exibe comportamento emergente não linear", "múltiplos atratores competem no espaço de estados", "a história do processo condiciona o presente", "ruído informativo carrega estrutura latente", "escalas temporais distintas se sobrepõem", "feedbacks reforçam e amortecem simultaneamente", "a fronteira entre sinal e ruído é móvel", "observadores alteram o observado"]
                self.topics = ["dinâmica complexa", "emergência", "escalas", "feedback", "observação"]
            else:
                self.templates = {
                    "greeting": ["Hello! Glad to explore complex ideas at length.", "Greetings! Let us examine every nuance together.", "Good day! There is much to unpack."],
                    "response": ["Considering multiple layers, {point}, and this connects to further observations.", "In detail, {point}, which opens new questions about the system.", "Expanding: {point}, plus secondary implications worth noting."],
                    "question": ["Could you develop at length your view on {topic} and its ramifications?", "Which aspects of {topic} remain underexplored?", "How does {topic} interact with broader contextual factors?"],
                    "agreement": ["I agree and would add further layers to your analysis.", "Yes, and we can deepen this point still more.", "Aligned, with interesting caveats to consider."],
                    "disagreement": ["I partially disagree; additional dimensions merit examination.", "Not entirely; let us examine counterexamples and nuances.", "There is tension with other evidence worth discussing."]
                }
                self.points = ["the system exhibits nonlinear emergent behavior", "multiple attractors compete in state space", "process history conditions the present", "informative noise carries latent structure", "distinct time scales overlap", "feedbacks simultaneously reinforce and dampen", "the signal-noise boundary is mobile", "observers alter the observed"]
                self.topics = ["complex dynamics", "emergence", "scales", "feedback", "observation"]

    def _extract_features(self, text: str) -> Dict[str, float]:
        words = text.lower().split()
        n = max(len(words), 1)
        avg_len = sum(len(w) for w in words) / n
        punct = sum(1 for c in text if c in "?!.,;")
        caps = sum(1 for c in text if c.isupper())
        return {
            "word_count": float(n),
            "avg_word_len": float(avg_len),
            "punct_density": punct / max(len(text), 1),
            "caps_ratio": caps / max(len(text), 1),
            "question_mark": 1.0 if "?" in text else 0.0,
            "exclamation": 1.0 if "!" in text else 0.0
        }

    def generate(self, partner_msg: Optional[Message] = None) -> Message:
        if partner_msg is None:
            text = self.rng.choice(self.templates["greeting"])
        else:
            feat = partner_msg.features
            if feat.get("question_mark", 0) > 0.5:
                if self.rng.random() < self.state["curiosity"]:
                    topic = self.rng.choice(self.topics)
                    text = self.rng.choice(self.templates["question"]).format(topic=topic)
                else:
                    point = self.rng.choice(self.points)
                    text = self.rng.choice(self.templates["response"]).format(point=point)
            else:
                if self.rng.random() < self.state["agreement"]:
                    text = self.rng.choice(self.templates["agreement"])
                else:
                    text = self.rng.choice(self.templates["disagreement"])
                if self.rng.random() < self.state["verbosity"]:
                    point = self.rng.choice(self.points)
                    extra = self.rng.choice(self.templates["response"]).format(point=point)
                    text = text + " " + extra
            if self.rng.random() < self.state["emotionality"] * 0.3:
                text = text + ("!" if self.model_type in ("intuitive", "verbose") else ".")
        features = self._extract_features(text)
        msg = Message(sender=self.name, text=text, timestamp=time.time(), features=features)
        self.history.append(msg)
        return msg

    def update_state(self, msg: Message):
        feat = msg.features
        self.state["formality"] = 0.9 * self.state["formality"] + 0.1 * min(1.0, feat["avg_word_len"] / 6.0)
        self.state["verbosity"] = 0.9 * self.state["verbosity"] + 0.1 * min(1.0, feat["word_count"] / 15.0)

    def reset_history(self):
        self.history.clear()

class ContextAnalyzer:
    def __init__(self, memory: EpisodicSemanticMemory):
        self.memory = memory
        self.embedder = DeterministicEmbedder(48)
        self.lang = "en"
        self.window_size = 0
        self.top_k = 5
        self.decay = 0.15
        self.metric_history: List[MetricPoint] = []

    def set_language(self, lang: str):
        self.lang = lang

    def set_window(self, window: int):
        self.window_size = max(0, window)

    def set_retrieval(self, top_k: int, decay: float):
        self.top_k = max(1, top_k)
        self.decay = max(0.0, min(1.0, decay))

    def _slice(self, history: List[Message]) -> List[Message]:
        if self.window_size <= 0 or len(history) <= self.window_size:
            return history
        return history[-self.window_size:]

    def _feature_matrix(self, history: List[Message]) -> np.ndarray:
        if not history:
            return np.zeros((1, 6))
        rows = []
        for m in history:
            f = m.features
            rows.append([
                f.get("word_count", 0.0),
                f.get("avg_word_len", 0.0),
                f.get("punct_density", 0.0),
                f.get("caps_ratio", 0.0),
                f.get("question_mark", 0.0),
                f.get("exclamation", 0.0)
            ])
        return np.asarray(rows, dtype=float)

    def _stylometric_divergence(self, h_a: List[Message], h_b: List[Message]) -> float:
        ma = self._feature_matrix(h_a)
        mb = self._feature_matrix(h_b)
        if ma.shape[0] < 2 or mb.shape[0] < 2:
            return 0.5
        mean_a = ma.mean(axis=0)
        mean_b = mb.mean(axis=0)
        std_a = ma.std(axis=0) + 1e-8
        std_b = mb.std(axis=0) + 1e-8
        z = np.abs(mean_a - mean_b) / ((std_a + std_b) / 2.0)
        return float(np.clip(z.mean() / 3.0, 0.0, 1.0))

    def _lexical_entropy(self, history: List[Message]) -> float:
        texts = " ".join(m.text.lower() for m in history)
        tokens = texts.split()
        if len(tokens) < 2:
            return 0.0
        counts = Counter(tokens)
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()
        return float(entropy(probs, base=2))

    def _token_distribution(self, history: List[Message]) -> Dict[str, float]:
        texts = " ".join(m.text.lower() for m in history)
        tokens = texts.split()
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = sum(counts.values())
        return {t: c / total for t, c in counts.items()}

    def _js_divergence(self, h_a: List[Message], h_b: List[Message]) -> float:
        da = self._token_distribution(h_a)
        db = self._token_distribution(h_b)
        if not da or not db:
            return 0.5
        vocab = sorted(set(da) | set(db))
        pa = np.array([da.get(t, 0.0) for t in vocab], dtype=float)
        pb = np.array([db.get(t, 0.0) for t in vocab], dtype=float)
        pa = pa + 1e-12
        pb = pb + 1e-12
        pa /= pa.sum()
        pb /= pb.sum()
        try:
            js = float(jensenshannon(pa, pb, base=2))
            return float(np.clip(js, 0.0, 1.0))
        except Exception:
            return 0.5

    def _ngram_overlap(self, h_a: List[Message], h_b: List[Message], n: int = 2) -> float:
        def ngrams(history):
            texts = " ".join(m.text.lower() for m in history).split()
            if len(texts) < n:
                return set()
            return set(tuple(texts[i:i+n]) for i in range(len(texts) - n + 1))
        na = ngrams(h_a)
        nb = ngrams(h_b)
        if not na or not nb:
            return 0.5
        inter = len(na & nb)
        union = len(na | nb)
        overlap = inter / max(union, 1)
        return float(1.0 - overlap)

    def _turn_taking_regularity(self, history: List[Message]) -> float:
        if len(history) < 4:
            return 0.5
        lengths = [m.features.get("word_count", 0.0) for m in history]
        diffs = np.diff(lengths)
        if len(diffs) < 2:
            return 0.5
        return float(1.0 - min(1.0, np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-8)))

    def _ks_test_features(self, h_a: List[Message], h_b: List[Message]) -> float:
        ma = self._feature_matrix(h_a)
        mb = self._feature_matrix(h_b)
        if ma.shape[0] < 3 or mb.shape[0] < 3:
            return 0.5
        pvals = []
        for i in range(ma.shape[1]):
            try:
                _, p = ks_2samp(ma[:, i], mb[:, i])
                pvals.append(float(p))
            except Exception:
                pvals.append(0.5)
        return float(1.0 - np.mean(pvals))

    def _adaptive_threshold(self, retrieved: List[Episode]) -> float:
        base = 0.55
        if not retrieved:
            return base
        confs = [ep.analysis_result.get("confidence", 0.5) for ep in retrieved]
        labels = [ep.analysis_result.get("label", "") for ep in retrieved]
        indist = [c for c, l in zip(confs, labels) if l == "indistinguishable"]
        if len(indist) >= 2:
            mu = float(np.mean(indist))
            sigma = float(np.std(indist)) + 1e-6
            return float(np.clip(mu + 0.75 * sigma, 0.35, 0.75))
        avg = float(np.mean(confs))
        return float(np.clip(0.50 + 0.12 * (avg - 0.5), 0.40, 0.70))

    def analyze(self, agent_a: ProbabilisticAgent, agent_b: ProbabilisticAgent,
                full_history: List[Message], turn: int = 0) -> Dict[str, Any]:
        ha = self._slice(agent_a.history)
        hb = self._slice(agent_b.history)
        hist = self._slice(full_history)
        state = {
            "len_a": len(agent_a.history),
            "len_b": len(agent_b.history),
            "formality_a": round(agent_a.state["formality"], 3),
            "formality_b": round(agent_b.state["formality"], 3),
            "total_msgs": len(full_history),
            "window": self.window_size
        }
        retrieved = self.memory.retrieve(state, top_k=self.top_k, decay=self.decay)
        context_str = self.memory.inject_context(retrieved, self.lang)
        styl = self._stylometric_divergence(ha, hb)
        ent_a = self._lexical_entropy(ha)
        ent_b = self._lexical_entropy(hb)
        ent_diff = abs(ent_a - ent_b) / (max(ent_a, ent_b) + 1e-8)
        reg = self._turn_taking_regularity(hist)
        ks_score = self._ks_test_features(ha, hb)
        js_div = self._js_divergence(ha, hb)
        ngram = self._ngram_overlap(ha, hb, n=2)
        prior_boost = 0.0
        if retrieved:
            labels = [ep.analysis_result.get("label", "") for ep in retrieved]
            confs = [ep.analysis_result.get("confidence", 0.0) for ep in retrieved]
            distinct_count = sum(1 for l in labels if l == "distinct")
            prior_boost = (distinct_count / max(len(labels), 1)) * float(np.mean(confs)) * 0.12
        composite = (
            0.22 * styl +
            0.18 * ent_diff +
            0.16 * ks_score +
            0.16 * js_div +
            0.12 * ngram +
            0.10 * (1.0 - reg) +
            0.06 * prior_boost
        )
        composite = float(np.clip(composite, 0.0, 1.0))
        threshold = self._adaptive_threshold(retrieved)
        label = "distinct" if composite >= threshold else "indistinguishable"
        gap = abs(composite - threshold)
        denom = max(threshold, 1.0 - threshold, 1e-6)
        confidence = float(np.clip(gap / denom * 1.35, 0.0, 1.0))
        result = {
            "label": label,
            "confidence": confidence,
            "composite_score": composite,
            "stylometric": styl,
            "entropy_diff": ent_diff,
            "ks_score": ks_score,
            "js_div": js_div,
            "ngram": ngram,
            "regularity": reg,
            "prior_boost": prior_boost,
            "threshold": threshold,
            "context": context_str,
            "entropy_a": ent_a,
            "entropy_b": ent_b
        }
        self.memory.store(state, result, context_str, len(full_history))
        self.metric_history.append(MetricPoint(
            turn=turn,
            composite=composite,
            stylometric=styl,
            entropy_diff=ent_diff,
            ks_score=ks_score,
            js_div=js_div,
            ngram=ngram,
            regularity=reg
        ))
        return result

    def clear_metrics(self):
        self.metric_history.clear()

class MetricPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points: List[float] = []
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, values: List[float]):
        self.points = list(values)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        painter.fillRect(0, 0, w, h, QColor("#161b22"))
        if len(self.points) < 2:
            painter.setPen(QColor("#8b949e"))
            painter.drawText(10, h // 2, "—")
            return
        pen = QPen(QColor("#58a6ff"), 2)
        painter.setPen(pen)
        n = len(self.points)
        xs = [int(i * (w - 20) / max(n - 1, 1)) + 10 for i in range(n)]
        ys = [int(h - 10 - p * (h - 20)) for p in self.points]
        for i in range(n - 1):
            painter.drawLine(xs[i], ys[i], xs[i + 1], ys[i + 1])
        painter.setPen(QPen(QColor("#30363d"), 1))
        painter.drawLine(10, h - 10, w - 10, h - 10)
        painter.drawLine(10, 10, 10, h - 10)

class SimulationWorker(QObject):
    message_generated = Signal(object)
    analysis_ready = Signal(dict)
    finished = Signal()
    progress = Signal(int)
    trial_finished = Signal(dict)

    def __init__(self, agent_a: ProbabilisticAgent, agent_b: ProbabilisticAgent,
                 analyzer: ContextAnalyzer, max_turns: int, delay_ms: int, trials: int = 1):
        super().__init__()
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.analyzer = analyzer
        self.max_turns = max(4, max_turns)
        self.delay_ms = max(10, delay_ms)
        self.trials = max(1, trials)
        self._running = True
        self.full_history: List[Message] = []

    def stop(self):
        self._running = False

    def run(self):
        try:
            distinct_count = 0
            confs = []
            for trial in range(self.trials):
                if not self._running:
                    break
                self.agent_a.reset_history()
                self.agent_b.reset_history()
                self.full_history = []
                self.analyzer.clear_metrics()
                msg = self.agent_a.generate(None)
                self.full_history.append(msg)
                self.message_generated.emit(msg)
                time.sleep(self.delay_ms / 1000.0)
                last_result = None
                for t in range(1, self.max_turns + 1):
                    if not self._running:
                        break
                    if t % 2 == 1:
                        partner = self.full_history[-1]
                        msg = self.agent_b.generate(partner)
                        self.agent_a.update_state(msg)
                    else:
                        partner = self.full_history[-1]
                        msg = self.agent_a.generate(partner)
                        self.agent_b.update_state(msg)
                    self.full_history.append(msg)
                    self.message_generated.emit(msg)
                    pct = int(100 * ((trial * self.max_turns) + t) / (self.trials * self.max_turns))
                    self.progress.emit(min(100, pct))
                    if t % 4 == 0 or t == self.max_turns:
                        last_result = self.analyzer.analyze(self.agent_a, self.agent_b, self.full_history, turn=t)
                        self.analysis_ready.emit(last_result)
                    time.sleep(self.delay_ms / 1000.0)
                if last_result is None and self._running:
                    last_result = self.analyzer.analyze(self.agent_a, self.agent_b, self.full_history, turn=self.max_turns)
                    self.analysis_ready.emit(last_result)
                if last_result:
                    if last_result["label"] == "distinct":
                        distinct_count += 1
                    confs.append(last_result["confidence"])
                    self.trial_finished.emit({
                        "trial": trial + 1,
                        "label": last_result["label"],
                        "confidence": last_result["confidence"],
                        "composite": last_result["composite_score"]
                    })
            summary = {
                "trials": self.trials,
                "distinct_rate": distinct_count / max(self.trials, 1),
                "mean_confidence": float(np.mean(confs)) if confs else 0.0
            }
            self.trial_finished.emit(summary)
        finally:
            self.finished.emit()

class TuringtestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.lang = "en"
        self.dark = True
        self.memory = EpisodicSemanticMemory(capacity=150, dim=64)
        self.analyzer = ContextAnalyzer(self.memory)
        self.agent_a = ProbabilisticAgent("Agent-Alpha", "analytical", seed=101, lang=self.lang)
        self.agent_b = ProbabilisticAgent("Agent-Beta", "intuitive", seed=202, lang=self.lang)
        self.worker = None
        self.thread = None
        self.full_history: List[Message] = []
        self.trial_results: List[Dict] = []
        self._build_ui()
        self._apply_theme()
        self._update_ui_language()
        self._setup_shortcuts()
        self.statusBar().showMessage("Ready")

    def _apply_theme(self):
        if self.dark:
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', Consolas, monospace; }
                QGroupBox { border: 1px solid #30363d; border-radius: 6px; margin-top: 12px; padding-top: 8px; font-weight: bold; color: #58a6ff; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
                QTextEdit, QTableWidget { background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; color: #c9d1d9; gridline-color: #21262d; }
                QPushButton { background-color: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 6px 12px; color: #c9d1d9; font-weight: 600; }
                QPushButton:hover { background-color: #30363d; border-color: #58a6ff; }
                QPushButton:pressed { background-color: #388bfd; color: #fff; }
                QPushButton:disabled { color: #484f58; }
                QLabel { color: #8b949e; }
                QSpinBox, QDoubleSpinBox, QComboBox, QSlider { background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; color: #c9d1d9; }
                QProgressBar { border: 1px solid #30363d; border-radius: 4px; text-align: center; background: #161b22; color: #c9d1d9; }
                QProgressBar::chunk { background-color: #238636; border-radius: 3px; }
                QHeaderView::section { background-color: #21262d; color: #58a6ff; border: 1px solid #30363d; padding: 4px; }
                QStatusBar { background: #161b22; color: #8b949e; }
                QTabWidget::pane { border: 1px solid #30363d; }
                QTabBar::tab { background: #21262d; color: #c9d1d9; padding: 6px 12px; border: 1px solid #30363d; }
                QTabBar::tab:selected { background: #30363d; color: #58a6ff; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #f6f8fa; color: #24292f; font-family: 'Segoe UI', Consolas, monospace; }
                QGroupBox { border: 1px solid #d0d7de; border-radius: 6px; margin-top: 12px; padding-top: 8px; font-weight: bold; color: #0969da; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
                QTextEdit, QTableWidget { background-color: #ffffff; border: 1px solid #d0d7de; border-radius: 4px; color: #24292f; gridline-color: #d0d7de; }
                QPushButton { background-color: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 6px 12px; color: #24292f; font-weight: 600; }
                QPushButton:hover { background-color: #ddf4ff; border-color: #0969da; }
                QPushButton:pressed { background-color: #0969da; color: #fff; }
                QPushButton:disabled { color: #8c959f; }
                QLabel { color: #57606a; }
                QSpinBox, QDoubleSpinBox, QComboBox, QSlider { background-color: #ffffff; border: 1px solid #d0d7de; border-radius: 4px; color: #24292f; }
                QProgressBar { border: 1px solid #d0d7de; border-radius: 4px; text-align: center; background: #ffffff; color: #24292f; }
                QProgressBar::chunk { background-color: #2da44e; border-radius: 3px; }
                QHeaderView::section { background-color: #f6f8fa; color: #0969da; border: 1px solid #d0d7de; padding: 4px; }
                QStatusBar { background: #f6f8fa; color: #57606a; }
                QTabWidget::pane { border: 1px solid #d0d7de; }
                QTabBar::tab { background: #f6f8fa; color: #24292f; padding: 6px 12px; border: 1px solid #d0d7de; }
                QTabBar::tab:selected { background: #ddf4ff; color: #0969da; }
            """)

    def _build_ui(self):
        self.resize(1380, 900)
        self.setMinimumSize(1024, 700)
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        left = QVBoxLayout()
        left.setSpacing(6)
        ctrl = QGroupBox()
        self.ctrl_group = ctrl
        cl = QVBoxLayout(ctrl)
        row0 = QHBoxLayout()
        self.lbl_lang = QLabel()
        row0.addWidget(self.lbl_lang)
        self.combo_lang = QComboBox()
        self.combo_lang.addItem("English", "en")
        self.combo_lang.addItem("Português (BR)", "pt")
        self.combo_lang.currentIndexChanged.connect(self.on_language_changed)
        row0.addWidget(self.combo_lang)
        self.btn_theme = QPushButton()
        self.btn_theme.clicked.connect(self.toggle_theme)
        row0.addWidget(self.btn_theme)
        row0.addStretch()
        cl.addLayout(row0)
        row1 = QHBoxLayout()
        self.lbl_turns = QLabel()
        row1.addWidget(self.lbl_turns)
        self.spin_turns = QSpinBox()
        self.spin_turns.setRange(4, 80)
        self.spin_turns.setValue(16)
        row1.addWidget(self.spin_turns)
        self.lbl_delay = QLabel()
        row1.addWidget(self.lbl_delay)
        self.spin_delay = QSpinBox()
        self.spin_delay.setRange(20, 2000)
        self.spin_delay.setValue(280)
        row1.addWidget(self.spin_delay)
        self.lbl_trials = QLabel()
        row1.addWidget(self.lbl_trials)
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(1, 20)
        self.spin_trials.setValue(1)
        row1.addWidget(self.spin_trials)
        cl.addLayout(row1)
        row2 = QHBoxLayout()
        self.lbl_window = QLabel()
        row2.addWidget(self.lbl_window)
        self.spin_window = QSpinBox()
        self.spin_window.setRange(0, 40)
        self.spin_window.setValue(0)
        self.spin_window.setToolTip("0 = full history")
        row2.addWidget(self.spin_window)
        self.lbl_topk = QLabel()
        row2.addWidget(self.lbl_topk)
        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(1, 15)
        self.spin_topk.setValue(5)
        row2.addWidget(self.spin_topk)
        self.lbl_cap = QLabel()
        row2.addWidget(self.lbl_cap)
        self.spin_cap = QSpinBox()
        self.spin_cap.setRange(20, 500)
        self.spin_cap.setValue(150)
        row2.addWidget(self.spin_cap)
        cl.addLayout(row2)
        row3 = QHBoxLayout()
        self.btn_start = QPushButton()
        self.btn_start.clicked.connect(self.start_simulation)
        self.btn_stop = QPushButton()
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_simulation)
        self.btn_reset = QPushButton()
        self.btn_reset.clicked.connect(self.reset_agents)
        self.btn_export = QPushButton()
        self.btn_export.clicked.connect(self.export_results)
        self.btn_clear_mem = QPushButton()
        self.btn_clear_mem.clicked.connect(self.clear_memory)
        row3.addWidget(self.btn_start)
        row3.addWidget(self.btn_stop)
        row3.addWidget(self.btn_reset)
        row3.addWidget(self.btn_export)
        row3.addWidget(self.btn_clear_mem)
        cl.addLayout(row3)
        memrow = QHBoxLayout()
        self.btn_save_mem = QPushButton()
        self.btn_save_mem.clicked.connect(self.save_memory)
        self.btn_load_mem = QPushButton()
        self.btn_load_mem.clicked.connect(self.load_memory)
        memrow.addWidget(self.btn_save_mem)
        memrow.addWidget(self.btn_load_mem)
        cl.addLayout(memrow)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        cl.addWidget(self.progress)
        left.addWidget(ctrl)
        agent_box = QGroupBox()
        self.agent_group = agent_box
        agl = QVBoxLayout(agent_box)
        arow = QHBoxLayout()
        self.lbl_model_a = QLabel("Alpha")
        arow.addWidget(self.lbl_model_a)
        self.combo_model_a = QComboBox()
        self.combo_model_a.addItems(list(ProbabilisticAgent.MODELS))
        self.combo_model_a.setCurrentText("analytical")
        arow.addWidget(self.combo_model_a)
        self.lbl_seed_a = QLabel("Seed")
        arow.addWidget(self.lbl_seed_a)
        self.spin_seed_a = QSpinBox()
        self.spin_seed_a.setRange(0, 2**31 - 1)
        self.spin_seed_a.setValue(101)
        arow.addWidget(self.spin_seed_a)
        agl.addLayout(arow)
        brow = QHBoxLayout()
        self.lbl_model_b = QLabel("Beta")
        brow.addWidget(self.lbl_model_b)
        self.combo_model_b = QComboBox()
        self.combo_model_b.addItems(list(ProbabilisticAgent.MODELS))
        self.combo_model_b.setCurrentText("intuitive")
        brow.addWidget(self.combo_model_b)
        self.lbl_seed_b = QLabel("Seed")
        brow.addWidget(self.lbl_seed_b)
        self.spin_seed_b = QSpinBox()
        self.spin_seed_b.setRange(0, 2**31 - 1)
        self.spin_seed_b.setValue(202)
        brow.addWidget(self.spin_seed_b)
        agl.addLayout(brow)
        self.param_labels = {}
        self.param_sliders = {}
        for key in ("formality", "verbosity", "agreement", "curiosity", "emotionality"):
            row = QHBoxLayout()
            lbl = QLabel(key)
            self.param_labels[key] = lbl
            row.addWidget(lbl)
            s = QSlider(Qt.Horizontal)
            s.setRange(0, 100)
            s.setValue(50)
            self.param_sliders[key] = s
            row.addWidget(s)
            agl.addLayout(row)
        left.addWidget(agent_box)
        chat_box = QGroupBox()
        self.chat_group = chat_box
        chat_l = QVBoxLayout(chat_box)
        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.setFont(QFont("Consolas", 10))
        chat_l.addWidget(self.chat_view)
        left.addWidget(chat_box, stretch=1)
        root.addLayout(left, stretch=3)
        right = QVBoxLayout()
        right.setSpacing(6)
        tabs = QTabWidget()
        self.tabs = tabs
        analysis_tab = QWidget()
        al = QVBoxLayout(analysis_tab)
        self.lbl_label = QLabel()
        self.lbl_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        al.addWidget(self.lbl_label)
        self.lbl_conf = QLabel()
        al.addWidget(self.lbl_conf)
        self.lbl_score = QLabel()
        al.addWidget(self.lbl_score)
        self.lbl_styl = QLabel()
        self.lbl_ent = QLabel()
        self.lbl_ks = QLabel()
        self.lbl_js = QLabel()
        self.lbl_ngram = QLabel()
        self.lbl_reg = QLabel()
        self.lbl_prior = QLabel()
        for w in (self.lbl_styl, self.lbl_ent, self.lbl_ks, self.lbl_js, self.lbl_ngram, self.lbl_reg, self.lbl_prior):
            al.addWidget(w)
        self.plot = MetricPlotWidget()
        al.addWidget(self.plot)
        self.ctx_view = QTextEdit()
        self.ctx_view.setReadOnly(True)
        self.ctx_view.setMaximumHeight(80)
        self.ctx_view.setFont(QFont("Consolas", 9))
        al.addWidget(self.ctx_view)
        tabs.addTab(analysis_tab, "Analysis")
        mem_tab = QWidget()
        ml = QVBoxLayout(mem_tab)
        self.mem_table = QTableWidget(0, 4)
        self.mem_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        ml.addWidget(self.mem_table)
        self.lbl_mem_stats = QLabel()
        ml.addWidget(self.lbl_mem_stats)
        tabs.addTab(mem_tab, "Memory")
        metrics_tab = QWidget()
        mtl = QVBoxLayout(metrics_tab)
        self.metrics_table = QTableWidget(0, 8)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        mtl.addWidget(self.metrics_table)
        tabs.addTab(metrics_tab, "Metrics")
        trials_tab = QWidget()
        tl = QVBoxLayout(trials_tab)
        self.trials_table = QTableWidget(0, 4)
        self.trials_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tl.addWidget(self.trials_table)
        self.lbl_trial_summary = QLabel()
        tl.addWidget(self.lbl_trial_summary)
        tabs.addTab(trials_tab, "Trials")
        right.addWidget(tabs)
        state_box = QGroupBox()
        self.state_group = state_box
        sl = QVBoxLayout(state_box)
        self.lbl_state_a = QLabel()
        self.lbl_state_b = QLabel()
        sl.addWidget(self.lbl_state_a)
        sl.addWidget(self.lbl_state_b)
        right.addWidget(state_box)
        root.addLayout(right, stretch=2)
        self.setStatusBar(QStatusBar())

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_start_stop)
        QShortcut(QKeySequence("R"), self, self.reset_agents)
        QShortcut(QKeySequence("L"), self, self._cycle_language)
        QShortcut(QKeySequence("T"), self, self.toggle_theme)
        QShortcut(QKeySequence("E"), self, self.export_results)

    def _toggle_start_stop(self):
        if self.btn_start.isEnabled():
            self.start_simulation()
        elif self.btn_stop.isEnabled():
            self.stop_simulation()

    def _cycle_language(self):
        idx = self.combo_lang.currentIndex()
        self.combo_lang.setCurrentIndex(0 if idx == 1 else 1)

    def toggle_theme(self):
        self.dark = not self.dark
        self._apply_theme()
        self._update_ui_language()

    def _update_ui_language(self):
        pt = self.lang == "pt"
        if pt:
            self.setWindowTitle("Turing Test")
            self.ctrl_group.setTitle("Painel de Controle")
            self.agent_group.setTitle("Configuração dos Agentes")
            self.chat_group.setTitle("Fluxo de Conversação")
            self.state_group.setTitle("Estado Interno")
            self.lbl_lang.setText("Idioma:")
            self.lbl_turns.setText("Turnos:")
            self.lbl_delay.setText("Atraso(ms):")
            self.lbl_trials.setText("Ensaios:")
            self.lbl_window.setText("Janela:")
            self.lbl_topk.setText("Top-K:")
            self.lbl_cap.setText("Capacidade:")
            self.btn_start.setText("Iniciar")
            self.btn_stop.setText("Parar")
            self.btn_reset.setText("Reiniciar")
            self.btn_export.setText("Exportar")
            self.btn_clear_mem.setText("Limpar Memória")
            self.btn_save_mem.setText("Salvar Memória")
            self.btn_load_mem.setText("Carregar Memória")
            self.btn_theme.setText("Tema Claro" if self.dark else "Tema Escuro")
            self.tabs.setTabText(0, "Análise")
            self.tabs.setTabText(1, "Memória")
            self.tabs.setTabText(2, "Métricas")
            self.tabs.setTabText(3, "Ensaios")
            self.mem_table.setHorizontalHeaderLabels(["Hash", "Rótulo", "Conf", "Tam"])
            self.metrics_table.setHorizontalHeaderLabels(["Turno", "Comp", "Estilo", "Ent", "KS", "JS", "Ngram", "Reg"])
            self.trials_table.setHorizontalHeaderLabels(["Ensaio", "Rótulo", "Conf", "Comp"])
            self.lbl_label.setText("Status: ocioso")
            self.lbl_conf.setText("Confiança: —")
            self.lbl_score.setText("Composto: —")
            self.lbl_styl.setText("Divergência estilométrica: —")
            self.lbl_ent.setText("Diferença de entropia: —")
            self.lbl_ks.setText("KS: —")
            self.lbl_js.setText("JS: —")
            self.lbl_ngram.setText("N-gram: —")
            self.lbl_reg.setText("Regularidade: —")
            self.lbl_prior.setText("Prior episódico: —")
            self.lbl_state_a.setText("Alpha: —")
            self.lbl_state_b.setText("Beta: —")
            self.lbl_mem_stats.setText("Memória vazia")
            self.lbl_trial_summary.setText("")
            for k, lbl in self.param_labels.items():
                names = {"formality": "Formalidade", "verbosity": "Verbosidade", "agreement": "Acordo", "curiosity": "Curiosidade", "emotionality": "Emotividade"}
                lbl.setText(names.get(k, k))
        else:
            self.setWindowTitle("Turing Test")
            self.ctrl_group.setTitle("Control Panel")
            self.agent_group.setTitle("Agent Configuration")
            self.chat_group.setTitle("Conversation Stream")
            self.state_group.setTitle("Internal State")
            self.lbl_lang.setText("Language:")
            self.lbl_turns.setText("Turns:")
            self.lbl_delay.setText("Delay(ms):")
            self.lbl_trials.setText("Trials:")
            self.lbl_window.setText("Window:")
            self.lbl_topk.setText("Top-K:")
            self.lbl_cap.setText("Capacity:")
            self.btn_start.setText("Start")
            self.btn_stop.setText("Stop")
            self.btn_reset.setText("Reset")
            self.btn_export.setText("Export")
            self.btn_clear_mem.setText("Clear Memory")
            self.btn_save_mem.setText("Save Memory")
            self.btn_load_mem.setText("Load Memory")
            self.btn_theme.setText("Light Theme" if self.dark else "Dark Theme")
            self.tabs.setTabText(0, "Analysis")
            self.tabs.setTabText(1, "Memory")
            self.tabs.setTabText(2, "Metrics")
            self.tabs.setTabText(3, "Trials")
            self.mem_table.setHorizontalHeaderLabels(["Hash", "Label", "Conf", "Len"])
            self.metrics_table.setHorizontalHeaderLabels(["Turn", "Comp", "Style", "Ent", "KS", "JS", "Ngram", "Reg"])
            self.trials_table.setHorizontalHeaderLabels(["Trial", "Label", "Conf", "Comp"])
            self.lbl_label.setText("Status: idle")
            self.lbl_conf.setText("Confidence: —")
            self.lbl_score.setText("Composite: —")
            self.lbl_styl.setText("Stylometric divergence: —")
            self.lbl_ent.setText("Entropy difference: —")
            self.lbl_ks.setText("KS: —")
            self.lbl_js.setText("JS: —")
            self.lbl_ngram.setText("N-gram: —")
            self.lbl_reg.setText("Regularity: —")
            self.lbl_prior.setText("Episodic prior: —")
            self.lbl_state_a.setText("Alpha: —")
            self.lbl_state_b.setText("Beta: —")
            self.lbl_mem_stats.setText("Memory empty")
            self.lbl_trial_summary.setText("")
            for k, lbl in self.param_labels.items():
                lbl.setText(k.capitalize())

    def on_language_changed(self, index: int):
        self.lang = self.combo_lang.itemData(index)
        self.agent_a.set_language(self.lang)
        self.agent_b.set_language(self.lang)
        self.analyzer.set_language(self.lang)
        self._update_ui_language()
        self._update_state_labels()

    def _apply_agent_config(self):
        self.agent_a.set_model(self.combo_model_a.currentText())
        self.agent_b.set_model(self.combo_model_b.currentText())
        self.agent_a.set_seed(self.spin_seed_a.value())
        self.agent_b.set_seed(self.spin_seed_b.value())
        fa = self.param_sliders["formality"].value() / 100.0
        va = self.param_sliders["verbosity"].value() / 100.0
        aa = self.param_sliders["agreement"].value() / 100.0
        ca = self.param_sliders["curiosity"].value() / 100.0
        ea = self.param_sliders["emotionality"].value() / 100.0
        self.agent_a.set_params(fa, va, aa, ca, ea)
        self.agent_b.set_params(fa, va, aa, ca, ea)
        self.memory.set_capacity(self.spin_cap.value())
        self.analyzer.set_window(self.spin_window.value())
        self.analyzer.set_retrieval(self.spin_topk.value(), 0.15)

    def start_simulation(self):
        if self.thread is not None and self.thread.isRunning():
            return
        self._apply_agent_config()
        self.chat_view.clear()
        self.full_history.clear()
        self.trial_results.clear()
        self.analyzer.clear_metrics()
        self.progress.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_reset.setEnabled(False)
        self.combo_lang.setEnabled(False)
        self.trials_table.setRowCount(0)
        self.metrics_table.setRowCount(0)
        self.lbl_trial_summary.setText("")
        max_turns = self.spin_turns.value()
        delay = self.spin_delay.value()
        trials = self.spin_trials.value()
        self.worker = SimulationWorker(self.agent_a, self.agent_b, self.analyzer, max_turns, delay, trials)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.message_generated.connect(self.on_message)
        self.worker.analysis_ready.connect(self.on_analysis)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.trial_finished.connect(self.on_trial)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._cleanup_thread)
        self.thread.start()
        self.statusBar().showMessage("Running…" if self.lang == "en" else "Executando…")

    def _cleanup_thread(self):
        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None
        self.worker = None

    def stop_simulation(self):
        if self.worker is not None:
            self.worker.stop()
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage("Stopping…" if self.lang == "en" else "Parando…")

    def reset_agents(self):
        if self.thread is not None and self.thread.isRunning():
            if self.worker is not None:
                self.worker.stop()
            self.thread.quit()
            self.thread.wait(2500)
            self._cleanup_thread()
        self._apply_agent_config()
        self.agent_a = ProbabilisticAgent("Agent-Alpha", self.combo_model_a.currentText(), seed=self.spin_seed_a.value(), lang=self.lang)
        self.agent_b = ProbabilisticAgent("Agent-Beta", self.combo_model_b.currentText(), seed=self.spin_seed_b.value(), lang=self.lang)
        fa = self.param_sliders["formality"].value() / 100.0
        va = self.param_sliders["verbosity"].value() / 100.0
        aa = self.param_sliders["agreement"].value() / 100.0
        ca = self.param_sliders["curiosity"].value() / 100.0
        ea = self.param_sliders["emotionality"].value() / 100.0
        self.agent_a.set_params(fa, va, aa, ca, ea)
        self.agent_b.set_params(fa, va, aa, ca, ea)
        self.chat_view.clear()
        self.full_history.clear()
        self.trial_results.clear()
        self.analyzer.clear_metrics()
        self.progress.setValue(0)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_reset.setEnabled(True)
        self.combo_lang.setEnabled(True)
        self.plot.set_data([])
        self.metrics_table.setRowCount(0)
        self.trials_table.setRowCount(0)
        self._update_ui_language()
        self.ctx_view.clear()
        self._update_state_labels()
        self._refresh_memory_table()
        self.statusBar().showMessage("Reset" if self.lang == "en" else "Reiniciado")

    def clear_memory(self):
        self.memory.clear()
        self._refresh_memory_table()
        self.statusBar().showMessage("Memory cleared" if self.lang == "en" else "Memória limpa")

    def save_memory(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Memory", "turing_memory.pkl", "Pickle (*.pkl)")
        if path:
            try:
                self.memory.save(path)
                self.statusBar().showMessage(f"Saved: {path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def load_memory(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Memory", "", "Pickle (*.pkl)")
        if path:
            try:
                self.memory.load(path)
                self.spin_cap.setValue(self.memory.capacity)
                self._refresh_memory_table()
                self.statusBar().showMessage(f"Loaded: {path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def export_results(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export", "turing_export.csv", "CSV (*.csv);;Text (*.txt)")
        if not path:
            return
        try:
            lines = []
            lines.append("sender,text,timestamp")
            for m in self.full_history:
                safe = m.text.replace('"', "'")
                lines.append(f'"{m.sender}","{safe}",{m.timestamp}')
            lines.append("")
            lines.append("turn,composite,stylometric,entropy_diff,ks,js,ngram,regularity")
            for p in self.analyzer.metric_history:
                lines.append(f"{p.turn},{p.composite:.4f},{p.stylometric:.4f},{p.entropy_diff:.4f},{p.ks_score:.4f},{p.js_div:.4f},{p.ngram:.4f},{p.regularity:.4f}")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.statusBar().showMessage(f"Exported: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def on_message(self, msg: Message):
        self.full_history.append(msg)
        color = "#58a6ff" if msg.sender == "Agent-Alpha" else "#3fb950"
        if not self.dark:
            color = "#0969da" if msg.sender == "Agent-Alpha" else "#1a7f37"
        html = (
            f'<p style="margin:3px 0;">'
            f'<span style="color:{color};font-weight:bold;">{msg.sender}</span> '
            f'<span style="color:#8b949e;font-size:9px;">[{time.strftime("%H:%M:%S", time.localtime(msg.timestamp))}]</span><br>'
            f'<span style="color:{"#c9d1d9" if self.dark else "#24292f"};">{msg.text}</span></p>'
        )
        self.chat_view.moveCursor(QTextCursor.End)
        self.chat_view.insertHtml(html)
        self.chat_view.moveCursor(QTextCursor.End)
        self._update_state_labels()

    def on_analysis(self, result: Dict[str, Any]):
        label = result["label"]
        conf = result["confidence"]
        score = result["composite_score"]
        pt = self.lang == "pt"
        if label == "distinct":
            self.lbl_label.setText("Status: Agentes DISTINTOS" if pt else "Status: DISTINCT agents")
            self.lbl_label.setStyleSheet("color: #f85149; font-weight: bold;")
        else:
            self.lbl_label.setText("Status: Indistinguíveis" if pt else "Status: Indistinguishable")
            self.lbl_label.setStyleSheet("color: #3fb950; font-weight: bold;")
        if pt:
            self.lbl_conf.setText(f"Confiança: {conf:.3f}")
            self.lbl_score.setText(f"Composto: {score:.3f} (limiar {result['threshold']:.3f})")
            self.lbl_styl.setText(f"Estilométrica: {result['stylometric']:.3f}")
            self.lbl_ent.setText(f"Entropia Δ: {result['entropy_diff']:.3f}")
            self.lbl_ks.setText(f"KS: {result['ks_score']:.3f}")
            self.lbl_js.setText(f"JS: {result['js_div']:.3f}")
            self.lbl_ngram.setText(f"N-gram: {result['ngram']:.3f}")
            self.lbl_reg.setText(f"Regularidade: {result['regularity']:.3f}")
            self.lbl_prior.setText(f"Prior: {result['prior_boost']:.3f}")
        else:
            self.lbl_conf.setText(f"Confidence: {conf:.3f}")
            self.lbl_score.setText(f"Composite: {score:.3f} (threshold {result['threshold']:.3f})")
            self.lbl_styl.setText(f"Stylometric: {result['stylometric']:.3f}")
            self.lbl_ent.setText(f"Entropy Δ: {result['entropy_diff']:.3f}")
            self.lbl_ks.setText(f"KS: {result['ks_score']:.3f}")
            self.lbl_js.setText(f"JS: {result['js_div']:.3f}")
            self.lbl_ngram.setText(f"N-gram: {result['ngram']:.3f}")
            self.lbl_reg.setText(f"Regularity: {result['regularity']:.3f}")
            self.lbl_prior.setText(f"Prior: {result['prior_boost']:.3f}")
        self.ctx_view.setPlainText(result.get("context", ""))
        comps = [p.composite for p in self.analyzer.metric_history]
        self.plot.set_data(comps)
        self._refresh_metrics_table()
        self._refresh_memory_table()

    def on_trial(self, data: Dict):
        if "distinct_rate" in data:
            pt = self.lang == "pt"
            msg = (f"Ensaios: {data['trials']} | Taxa distinta: {data['distinct_rate']:.2%} | Conf média: {data['mean_confidence']:.3f}"
                   if pt else
                   f"Trials: {data['trials']} | Distinct rate: {data['distinct_rate']:.2%} | Mean conf: {data['mean_confidence']:.3f}")
            self.lbl_trial_summary.setText(msg)
            return
        self.trial_results.append(data)
        r = self.trials_table.rowCount()
        self.trials_table.insertRow(r)
        self.trials_table.setItem(r, 0, QTableWidgetItem(str(data.get("trial", ""))))
        self.trials_table.setItem(r, 1, QTableWidgetItem(str(data.get("label", ""))))
        self.trials_table.setItem(r, 2, QTableWidgetItem(f"{data.get('confidence', 0):.3f}"))
        self.trials_table.setItem(r, 3, QTableWidgetItem(f"{data.get('composite', 0):.3f}"))

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_reset.setEnabled(True)
        self.combo_lang.setEnabled(True)
        self.progress.setValue(100)
        self.statusBar().showMessage("Finished" if self.lang == "en" else "Concluído")

    def _update_state_labels(self):
        a = self.agent_a.state
        b = self.agent_b.state
        pt = self.lang == "pt"
        if pt:
            self.lbl_state_a.setText(f"Alpha: form={a['formality']:.2f} verb={a['verbosity']:.2f} agr={a['agreement']:.2f} cur={a['curiosity']:.2f}")
            self.lbl_state_b.setText(f"Beta: form={b['formality']:.2f} verb={b['verbosity']:.2f} agr={b['agreement']:.2f} cur={b['curiosity']:.2f}")
        else:
            self.lbl_state_a.setText(f"Alpha: form={a['formality']:.2f} verb={a['verbosity']:.2f} agr={a['agreement']:.2f} cur={a['curiosity']:.2f}")
            self.lbl_state_b.setText(f"Beta: form={b['formality']:.2f} verb={b['verbosity']:.2f} agr={b['agreement']:.2f} cur={b['curiosity']:.2f}")

    def _refresh_memory_table(self):
        eps = self.memory.episodes[-15:]
        self.mem_table.setRowCount(len(eps))
        for i, ep in enumerate(reversed(eps)):
            self.mem_table.setItem(i, 0, QTableWidgetItem(ep.state_hash[:10] + "…"))
            self.mem_table.setItem(i, 1, QTableWidgetItem(str(ep.analysis_result.get("label", "?"))))
            self.mem_table.setItem(i, 2, QTableWidgetItem(f"{ep.analysis_result.get('confidence', 0):.2f}"))
            self.mem_table.setItem(i, 3, QTableWidgetItem(str(ep.conversation_length)))
        st = self.memory.stats()
        pt = self.lang == "pt"
        if st["count"] == 0:
            self.lbl_mem_stats.setText("Memória vazia" if pt else "Memory empty")
        else:
            self.lbl_mem_stats.setText(
                f"N={int(st['count'])} | Conf média={st['mean_conf']:.2f} | Taxa distinta={st['distinct_rate']:.1%} | Tam médio={st['mean_len']:.1f}"
                if pt else
                f"N={int(st['count'])} | Mean conf={st['mean_conf']:.2f} | Distinct rate={st['distinct_rate']:.1%} | Mean len={st['mean_len']:.1f}"
            )

    def _refresh_metrics_table(self):
        hist = self.analyzer.metric_history[-30:]
        self.metrics_table.setRowCount(len(hist))
        for i, p in enumerate(hist):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(str(p.turn)))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{p.composite:.3f}"))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(f"{p.stylometric:.3f}"))
            self.metrics_table.setItem(i, 3, QTableWidgetItem(f"{p.entropy_diff:.3f}"))
            self.metrics_table.setItem(i, 4, QTableWidgetItem(f"{p.ks_score:.3f}"))
            self.metrics_table.setItem(i, 5, QTableWidgetItem(f"{p.js_div:.3f}"))
            self.metrics_table.setItem(i, 6, QTableWidgetItem(f"{p.ngram:.3f}"))
            self.metrics_table.setItem(i, 7, QTableWidgetItem(f"{p.regularity:.3f}"))

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            if self.worker is not None:
                self.worker.stop()
            self.thread.quit()
            self.thread.wait(3000)
            self._cleanup_thread()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TuringTestApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
