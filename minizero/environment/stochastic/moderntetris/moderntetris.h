#pragma once

#include "engine/step.hpp"
#include "stochastic_env.h"
#include <array>
#include <cmath>
#include <deque>
#include <string>
#include <vector>

namespace minizero::env::moderntetris {

constexpr char kModernTetrisName[] = "moderntetris";
constexpr int kModernTetrisNumPlayer = 1;
constexpr int kModernTetrisBoardSize = 10;
constexpr int kModernTetrisBoardWidth = engine::BOARD_RIGHT - engine::BOARD_LEFT + 1;
constexpr int kModernTetrisBoardHeight = engine::BOARD_BOTTOM - engine::BOARD_TOP + 1;
constexpr int kModernTetrisActionSize = 11;
constexpr int kModernTetrisChanceEventSize = 1;
constexpr int kModernTetrisDiscreteValueSize = 601;

extern const std::array<std::string, kModernTetrisActionSize + 1> kModernTetrisActionName;
void initialize();

class ModernTetrisAction : public BaseAction {
public:
    ModernTetrisAction() : BaseAction() {}
    ModernTetrisAction(int action_id, Player player) : BaseAction(action_id, player) {}
    ModernTetrisAction(const std::vector<std::string>& action_string_args);

    Player nextPlayer() const override { return player_ == Player::kPlayer1 ? Player::kPlayerNone : Player::kPlayer1; }
    std::string toConsoleString() const override;
};

class ModernTetrisEnv : public StochasticEnv<ModernTetrisAction> {
public:
    ModernTetrisEnv() {}

    void reset() override { reset(utils::Random::randInt()); }
    void reset(int seed) override;
    bool act(const ModernTetrisAction& action, bool with_chance = true) override;
    bool act(const std::vector<std::string>& action_string_args, bool with_chance = true) override { return act(ModernTetrisAction(action_string_args), with_chance); }
    bool actChanceEvent(const ModernTetrisAction& action) override;
    bool actChanceEvent();

    std::vector<ModernTetrisAction> getLegalActions() const override;
    std::vector<ModernTetrisAction> getLegalChanceEvents() const override;
    float getChanceEventProbability(const ModernTetrisAction& action) const override;
    bool isLegalAction(const ModernTetrisAction& action) const override;
    bool isLegalChanceEvent(const ModernTetrisAction& action) const override;
    bool isTerminal() const override;

    int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
    int getRotateChanceEvent(int event_id, utils::Rotation rotation) const override { return event_id; }
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const ModernTetrisAction& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getChanceEventFeatures(const ModernTetrisAction& event, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    int getNumInputChannels() const override;
    int getNumActionFeatureChannels() const override { return kModernTetrisActionSize; }
    int getNumChanceEventFeatureChannels() const override { return kModernTetrisChanceEventSize; }
    int getInputChannelHeight() const override { return kModernTetrisBoardHeight; }
    int getInputChannelWidth() const override { return kModernTetrisBoardWidth; }
    int getHiddenChannelHeight() const override { return kModernTetrisBoardHeight; }
    int getHiddenChannelWidth() const override { return kModernTetrisBoardWidth; }
    int getPolicySize() const override { return kModernTetrisActionSize; }
    int getChanceEventSize() const override { return kModernTetrisChanceEventSize; }
    int getDiscreteValueSize() const override { return kModernTetrisDiscreteValueSize; }

    std::string toString() const override;
    std::string name() const override { return kModernTetrisName; }
    int getNumPlayer() const override { return kModernTetrisNumPlayer; }
    float getReward() const override { return reward_; }
    float getEvalScore(bool is_resign = false) const override { return total_reward_; }

private:
    static engine::step::Action toEngineAction(int action_id);
    static int toPieceIndex(engine::PieceType piece_type);
    static bool isOccupied(engine::Cell cell);
    static int getChannelCount(int preview_size, int history_length);
    bool isActionEffective(const ModernTetrisAction& action) const;
    std::vector<float> captureActivePiecePlane() const;
    void resetActivePieceHistory();
    void pushActivePieceHistory(const std::vector<float>& plane);
    void writeBoardFeatures(std::vector<float>& features) const;
    void writePieceFeatures(std::vector<float>& features, int channel_offset, engine::PieceType piece_type) const;
    void fillScalarPlane(std::vector<float>& features, int channel, float value) const;
    void fillOneHotPlane(std::vector<float>& features, int channel_offset, int size, int index) const;

private:
    engine::step::Context ctx_;
    std::deque<std::vector<float>> active_piece_history_;
    int reward_ = 0;
    int total_reward_ = 0;
};

class ModernTetrisEnvLoader : public StochasticEnvLoader<ModernTetrisAction, ModernTetrisEnv> {
public:
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getChanceEventFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getChance(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getValue(const int pos) const override { return toDiscreteValue(pos < static_cast<int>(action_pairs_.size()) ? utils::transformValue(calculateNStepValue(pos)) : 0.0f); }
    std::vector<float> getAfterstateValue(const int pos) const override { return toDiscreteValue(pos < static_cast<int>(action_pairs_.size()) ? utils::transformValue(calculateNStepValue(pos) - BaseEnvLoader::getReward(pos)[0]) : 0.0f); }
    std::vector<float> getReward(const int pos) const override { return toDiscreteValue(pos < static_cast<int>(action_pairs_.size()) ? utils::transformValue(BaseEnvLoader::getReward(pos)[0]) : 0.0f); }
    float getPriority(const int pos) const override { return std::fabs(calculateNStepValue(pos) - BaseEnvLoader::getValue(pos)[0]); }

    std::string name() const override { return kModernTetrisName; }
    int getPolicySize() const override { return kModernTetrisActionSize; }
    int getChanceEventSize() const override { return kModernTetrisChanceEventSize; }
    int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
    int getRotateChanceEvent(int event_id, utils::Rotation rotation) const override { return event_id; }

private:
    float calculateNStepValue(const int pos) const;
    std::vector<float> toDiscreteValue(float value) const;
};

} // namespace minizero::env::moderntetris
