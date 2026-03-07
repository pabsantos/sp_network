#!/bin/bash
# Script para compilar o artigo LaTeX com bibliografia

cd "$(dirname "$0")"

echo "Compilando artigo.tex..."
pdflatex -interaction=nonstopmode artigo.tex > /dev/null
biber artigo
pdflatex -interaction=nonstopmode artigo.tex > /dev/null
pdflatex -interaction=nonstopmode artigo.tex > /dev/null

echo "Compilação concluída! Arquivo: artigo.pdf"
ls -lh artigo.pdf
