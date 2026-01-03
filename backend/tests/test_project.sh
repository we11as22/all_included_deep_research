#!/bin/bash

# Test All-Included Deep Research Project

echo "🧪 Testing All-Included Deep Research Project"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Testing: $test_name... "
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo "📦 1. Project Structure"
echo "----------------------"

run_test "Backend directory exists" "[ -d backend ]"
run_test "Frontend directory exists" "[ -d frontend ]"
run_test "Docker directory exists" "[ -d docker ]"
run_test "Backend pyproject.toml exists" "[ -f backend/pyproject.toml ]"
run_test "Frontend package.json exists" "[ -f frontend/package.json ]"
run_test "Docker compose file exists" "[ -f docker/docker-compose.yml ]"

echo ""
echo "🐍 2. Backend Structure"
echo "----------------------"

run_test "Backend src directory" "[ -d backend/src ]"
run_test "API module" "[ -d backend/src/api ]"
run_test "Workflow module" "[ -d backend/src/workflow ]"
run_test "Memory module" "[ -d backend/src/memory ]"
run_test "Streaming module" "[ -d backend/src/streaming ]"
run_test "Config module" "[ -d backend/src/config ]"
run_test "Database module" "[ -d backend/src/database ]"

echo ""
echo "⚛️  3. Frontend Structure"
echo "------------------------"

run_test "Frontend src directory" "[ -d frontend/src ]"
run_test "App directory" "[ -d frontend/src/app ]"
run_test "Components directory" "[ -d frontend/src/components ]"
run_test "Lib directory" "[ -d frontend/src/lib ]"
run_test "Styles directory" "[ -d frontend/src/styles ]"
run_test "Tailwind config" "[ -f frontend/tailwind.config.ts ]"
run_test "TypeScript config" "[ -f frontend/tsconfig.json ]"

echo ""
echo "📄 4. Key Files"
echo "--------------"

run_test "Backend main app" "[ -f backend/src/api/app.py ]"
run_test "Workflow factory" "[ -f backend/src/workflow/factory.py ]"
run_test "Speed workflow" "[ -f backend/src/workflow/speed_research.py ]"
run_test "Balanced workflow" "[ -f backend/src/workflow/balanced_research.py ]"
run_test "Quality workflow" "[ -f backend/src/workflow/quality_research.py ]"
run_test "SSE streaming" "[ -f backend/src/streaming/sse.py ]"
run_test "Frontend main page" "[ -f frontend/src/app/page.tsx ]"
run_test "Frontend layout" "[ -f frontend/src/app/layout.tsx ]"
run_test "API client" "[ -f frontend/src/lib/api.ts ]"

echo ""
echo "🔧 5. Configuration Files"
echo "------------------------"

run_test "Backend .env.example" "[ -f backend/.env.example ]"
run_test "Docker .env.example" "[ -f docker/.env.example ]"
run_test "Alembic config" "[ -f backend/alembic.ini ]"
run_test "Backend Dockerfile" "[ -f backend/Dockerfile ]"
run_test "Frontend Dockerfile" "[ -f frontend/Dockerfile ]"

echo ""
echo "🚀 6. Scripts"
echo "------------"

run_test "Docker start script" "[ -f docker/start.sh ]"
run_test "Docker stop script" "[ -f docker/stop.sh ]"
run_test "Start script is executable" "[ -x docker/start.sh ]"
run_test "Stop script is executable" "[ -x docker/stop.sh ]"

echo ""
echo "📚 7. Documentation"
echo "------------------"

run_test "README exists" "[ -f README.md ]"
run_test "README has content" "[ -s README.md ]"

echo ""
echo "=============================================="
echo "📊 Test Summary"
echo "=============================================="
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo ""
    echo "🎉 Project is ready to use!"
    echo ""
    echo "Next steps:"
    echo "1. Configure backend/.env with your API keys"
    echo "2. Configure docker/.env with your passwords"
    echo "3. Run: cd docker && ./start.sh"
    echo "4. Open http://localhost:3000 in your browser"
    exit 0
else
    echo -e "${RED}❌ Some tests failed!${NC}"
    echo ""
    echo "Please check the failed tests above."
    exit 1
fi

