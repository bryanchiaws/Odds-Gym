import numpy as np
from gym import spaces
from .base import BaseOddsEnv


class BasePercentageOddsEnv(BaseOddsEnv):
    """Base class for sports betting environments with non fixed bet size.

    Creates an OpenAI Gym environment that supports betting a non fixed amount
    on a single outcome for a single game.

    The main difference between the BaseOddsEnv is that the action space is defined
    differently (to accommodate that non fixed bet size).

    .. versionadded:: 0.2.0
    .. versionchanged:: 0.4.5
        Name changed to "BasePercentageOddsEnv"

    Parameters
    ----------
    action_space : gym.spaces.Box of shape (N,), where N is the number of possible
        outcomes for the game.
        The indexes are the percentage of the current balance to place on matching
        outcome, so that action[i] is the bet percentage for outcome[i].

        .. versionchanged:: 0.5.0
            Change action space so that each outcome has it's own independent bet
            percentage.
        .. versionchanged:: 0.6.0
            Change action space bounds to [-1, 1] and rescale the action back
            inside the step method.
        .. versionchanged:: 0.7.0
            Reduce dimesionality of action space, deduce the action implicitly
            from the given percentages

        The rescaling an action :math:`(a, p_0, ..., p_{N-1})\\in\\text{action_space}, -1 \\leq a, p_0, ..., p_{N-1} \\leq 1`:

        .. math::
            \\begin{cases}
                a' = \\lfloor (a + 1) * (2^{N-1}) \\rfloor\\\\
                p_i' = |p_i|
            \\end{cases}

    """

    def __init__(self, main_df, odds_column_names, num_possible_outcomes=3, results=None, *args, **kwargs):
        super().__init__(main_df, odds_column_names, num_possible_outcomes, results, *args, **kwargs)
        ACTION_DIM = int(num_possible_outcomes)
        self.action_space = spaces.Box(low=np.array([0.0] * ACTION_DIM),
                                       high=np.array([1.0] * ACTION_DIM))

    # @override
    def step(self, action):
        """ action is the fraction of balance to bet on each outcome i.e. np.array([0.0813, 0.0176, 0.255])"""
        odds = self.get_odds()
        reward = 0
        done = False
        info = self.create_info(action)
        if self.balance < 0:  # no more money :-(
            done = True
        else:
            bets = self.get_bet(action)
            results = self.get_results()
            if self.legal_bet(bets):  # making sure agent has enough money for the bet
                bets, odds, results = bets.reshape((-1, self.ACTION_DIM)), odds.reshape((-1, self.ACTION_DIM)), results.reshape((-1, self.ACTION_DIM))
                reward = self.get_reward(bets, odds, results)
                self.balance += reward
                info.update(legal_bet=True)
            else:
                reward = -1 * np.sum(bets)
            info.update(results=results.argmax())
            info.update(reward=reward)
            self.current_step += 1
            if self.finish():
                done = True
                next = np.ones(shape=self.observation_space.shape)
            else:
                next = self.get_obs()
        info.update(done=done)
        return next, reward, done, info
    
    # @override
    def finish(self):
        return self.current_step == self._df.shape[0] or self.balance < 0
    
    # @override
    def legal_bet(self, bets):
        return np.sum(bets) <= self.balance and np.all(bets >= 0)
    
    # @override
    def get_bet(self, action):
        return np.round(action * self.balance, 2)
    
    # @override
    def get_reward(self, bets, odds, results):
        """ Calculates the reward

        Parameters
        ----------
        bet : array of shape (1, n_odds) -- how much (dollars) the agent is betting on each outcome
        odds: dataframe of shape (1, n_odds)
            A games with its betting odds.
        results : array of shape (1, n_odds)

        Returns
        -------
        reward : float
            The amount of reward returned after previous action
        """
        returns = np.sum(bets * odds * results)
        expenses = np.sum(bets)
        profit = returns - expenses      
        return profit
    
    # @override
    def create_info(self, action):
        return {'current_step': self.current_step, 'odds': self.get_odds(), 'bets': f'home: {action[0]} | draw: {action[1]} | away: {action[2]}',
                'results': None, 'reward': 0, 'balance': self.balance, 'legal_bet': False, 'done': False}
