#!/bin/bash
set -e

echo "######################################################################"
echo "# COMPREHENSIVE REAL LLM TESTING"
echo "# Testing all search workflows with actual OpenRouter API"
echo "######################################################################"

cd /app

echo ""
echo "▶️  TEST 1/3: Query Classifier"
python test_1_classifier.py
if [ $? -ne 0 ]; then
    echo "❌ Test 1 FAILED"
    exit 1
fi

echo ""
echo "▶️  TEST 2/3: Web Search Mode (Speed: 2 iterations)"
python test_2_web_search.py
if [ $? -ne 0 ]; then
    echo "❌ Test 2 FAILED"
    exit 1
fi

echo ""
echo "▶️  TEST 3/3: Deep Search Mode (Balanced: 6 iterations)"
python test_3_deep_search.py
if [ $? -ne 0 ]; then
    echo "❌ Test 3 FAILED"
    exit 1
fi

echo ""
echo "######################################################################"
echo "# ✅ ALL TESTS PASSED!"
echo "######################################################################"
echo ""
echo "Summary:"
echo "  ✓ Classifier correctly routes queries"
echo "  ✓ Web search (speed) works with 2 iterations"
echo "  ✓ Deep search (balanced) works with 6 iterations"
echo "  ✓ All agents produce proper citations"
echo "  ✓ Real LLM integration functioning"
echo ""
