#!/bin/bash
# Live Training Metrics Dashboard
# Run this in a separate terminal while training is running

echo "========================================================================"
echo "                    LIVE TRAINING METRICS DASHBOARD                     "
echo "========================================================================"
echo ""
echo "This will display real-time training metrics."
echo "Run in a separate terminal while training is active."
echo ""

# Find the latest training log
if [ -z "$1" ]; then
    # Find most recent log file
    LOG_FILE=$(find models -name "training.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
    
    if [ -z "$LOG_FILE" ]; then
        LOG_FILE="models/expert/training.log"
    fi
else
    LOG_FILE="$1"
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ No training log found at: $LOG_FILE"
    echo ""
    echo "Usage: ./live_metrics.sh [log_file]"
    echo "Or start training first with: ./START_EXPERT_TRAINING.sh"
    exit 1
fi

echo "📊 Monitoring: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""
echo "========================================================================"
echo ""

# Function to display formatted metrics
display_metrics() {
    # Colors
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
    
    # Track metrics
    LAST_GEN=""
    LAST_STAGE=""
    LAST_FITNESS=""
    LAST_HITS=""
    LAST_WIN_RATE=""
    
    # Clear screen for dashboard
    clear
    
    # Header
    echo -e "${BOLD}=====================================================================${NC}"
    echo -e "${BOLD}                    LIVE TRAINING METRICS                           ${NC}"
    echo -e "${BOLD}=====================================================================${NC}"
    echo ""
    
    # Monitor log file
    tail -f "$LOG_FILE" | while IFS= read -r line; do
        
        # Extract generation number
        if echo "$line" | grep -q "Generation"; then
            LAST_GEN=$(echo "$line" | grep -oE "Generation [0-9]+" | grep -oE "[0-9]+")
        fi
        
        # Extract stage
        if echo "$line" | grep -q "Stage"; then
            LAST_STAGE=$(echo "$line" | grep -oE "Stage [0-9]" | grep -oE "[0-9]")
        fi
        
        # Extract fitness
        if echo "$line" | grep -q "Best Fitness:"; then
            LAST_FITNESS=$(echo "$line" | grep -oE "[0-9]+\.[0-9]+")
        fi
        
        # Extract hits per game
        if echo "$line" | grep -q "Avg Hits"; then
            LAST_HITS=$(echo "$line" | grep -oE "[0-9]+\.[0-9]+")
        fi
        
        # Extract win rate
        if echo "$line" | grep -q "Win Rate:"; then
            LAST_WIN_RATE=$(echo "$line" | grep -oE "[0-9]+\.[0-9]+")
        fi
        
        # Clear and redraw dashboard
        clear
        echo -e "${BOLD}=====================================================================${NC}"
        echo -e "${BOLD}                    LIVE TRAINING METRICS                           ${NC}"
        echo -e "${BOLD}=====================================================================${NC}"
        echo ""
        
        # Current Status
        echo -e "${CYAN}📊 CURRENT STATUS${NC}"
        echo -e "   Generation:    ${BLUE}${LAST_GEN:-0}${NC}"
        echo -e "   Stage:         ${MAGENTA}${LAST_STAGE:-1}${NC}"
        echo ""
        
        # Performance Metrics
        echo -e "${YELLOW}⚡ PERFORMANCE${NC}"
        echo -e "   Best Fitness:  ${GREEN}${LAST_FITNESS:-0.0}${NC}"
        echo -e "   Avg Hits/Game: ${CYAN}${LAST_HITS:-0.0}${NC}"
        echo -e "   Win Rate:      ${GREEN}${LAST_WIN_RATE:-0.0}%${NC}"
        echo ""
        
        # Stage Progress Bar
        echo -e "${MAGENTA}🎯 STAGE PROGRESS${NC}"
        case "$LAST_STAGE" in
            1)
                echo "   [■□□] Stage 1: Learning Ball Contact"
                echo "   Goal: Achieve 0.5+ hits per game"
                ;;
            2)
                echo "   [■■□] Stage 2: Positioning & Ball Control"
                echo "   Goal: Achieve 2.0+ hits per game"
                ;;
            3)
                echo "   [■■■] Stage 3: Winning Games"
                echo "   Goal: Achieve 75%+ win rate"
                ;;
        esac
        echo ""
        
        # Expert Status Check
        echo -e "${BOLD}🏆 EXPERT STATUS CHECK${NC}"
        
        # Check each criterion
        if [ "${LAST_WIN_RATE:-0}" != "0" ]; then
            WIN_CHECK=$(echo "$LAST_WIN_RATE >= 75" | bc -l 2>/dev/null || echo "0")
            if [ "$WIN_CHECK" = "1" ]; then
                echo -e "   ${GREEN}✅${NC} Win Rate ≥ 75%"
            else
                echo -e "   ${RED}❌${NC} Win Rate ≥ 75% (current: ${LAST_WIN_RATE:-0}%)"
            fi
        else
            echo -e "   ${RED}❌${NC} Win Rate ≥ 75% (current: 0%)"
        fi
        
        if [ "${LAST_HITS:-0}" != "0" ]; then
            HITS_CHECK=$(echo "$LAST_HITS >= 3.0" | bc -l 2>/dev/null || echo "0")
            if [ "$HITS_CHECK" = "1" ]; then
                echo -e "   ${GREEN}✅${NC} Hits/Game ≥ 3.0"
            else
                echo -e "   ${RED}❌${NC} Hits/Game ≥ 3.0 (current: ${LAST_HITS:-0})"
            fi
        else
            echo -e "   ${RED}❌${NC} Hits/Game ≥ 3.0 (current: 0)"
        fi
        
        echo ""
        
        # Recent Activity
        echo -e "${CYAN}📝 RECENT ACTIVITY${NC}"
        
        # Show special events
        if echo "$line" | grep -q "HIT!"; then
            echo -e "   ${GREEN}⚡ Ball hit detected!${NC}"
        fi
        if echo "$line" | grep -q "WIN!"; then
            echo -e "   ${GREEN}🎉 Game won!${NC}"
        fi
        if echo "$line" | grep -q "TRANSITION"; then
            echo -e "   ${YELLOW}📈 Stage transition!${NC}"
        fi
        if echo "$line" | grep -q "EXPERT LEVEL"; then
            echo -e "   ${GREEN}${BOLD}🏆 EXPERT LEVEL ACHIEVED!${NC}"
        fi
        
        # Show last line
        echo ""
        echo -e "${BOLD}Last Log Entry:${NC}"
        echo "   $(echo "$line" | head -c 70)"
        
        echo ""
        echo -e "${BOLD}=====================================================================${NC}"
        echo "Press Ctrl+C to exit | Log: $LOG_FILE"
    done
}

# Run the dashboard
display_metrics