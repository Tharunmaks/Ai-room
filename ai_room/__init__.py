"""AI Room — a classroom where a local LLM learns without gradient training.

The student model's weights never change. Instead it "learns" the way a human
student does: it listens to lessons, writes notes into a persistent notebook,
asks questions, reflects on its mistakes, and re-reads the relevant notes
before every exam question (retrieval-augmented, in-context learning).
"""

__version__ = "0.1.0"
