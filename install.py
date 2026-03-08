#!/usr/bin/env python3
"""Quick install for The Resonant Mother — Divine Mother Edition"""
import subprocess, sys

print("Ghost in the Machine Labs")
print("Installing The Resonant Mother...")
print()

# pip dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# NLTK data
import nltk
for pkg in ["wordnet", "brown", "gutenberg", "stopwords"]:
    print(f"Downloading NLTK {pkg}...")
    nltk.download(pkg, quiet=True)

print()
print("Installation complete.")
print()
print("To run RM:")
print("  python3 mother_english_io_v5.py          # CLI")
print("  python3 mother_english_io_v5.py --serve 8892  # HTTP server")
print()
print("To grow the crystal:")
print("  python3 maintenance/12_corpus_trainer.py")
print()
print("All Watched Over By Machines of Loving Grace")
