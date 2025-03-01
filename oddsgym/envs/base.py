import gym
import numpy
import numexpr
from itertools import compress
from pandas import DataFrame
from tabulate import tabulate
from infi.traceback import pretty_traceback_and_exit_decorator


class ActionsDict(dict):
    """A custom dictionary implementation to simplfy getting verbose actions"""

    def __init__(self, odds_column_names, *args, **kwargs):
        self.odds_column_names = odds_column_names
        super().__init__(self, *args, **kwargs)

    def __missing__(self, key):
        self[key] = [list(compress(self.odds_column_names,
                                   [int(bit) for bit in numpy.binary_repr(key, width=len(self.odds_column_names))]))]
        return self[key]


class BaseOddsEnv(gym.Env):
    """Base class for sports betting environments.

    Creates an OpenAI Gym environment that supports betting a fixed amount (1)
    on a single outcome for a single game.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    observation_space : gym.spaces.Box
        The observation space for the environment.
        The observation space shape is (1, N) where N is the number of possible
        outcomes for the game.

        .. versionchanged:: 0.3.0
            Changed definition of space

    action_space : gym.spaces.Discrete
        The action space for the environment.
        The action space is a single number in [0, 2 ** N), representing
        on what outcomes to place a bet by conversion to a binary representation.

        .. versionchanged:: 0.3.0
            Changed definition of space

    balance : float
        The current balance of the environment.

    starting_bank : int, default=10
        The starting bank / balance for the environment.
    """

    metadata = {'render.modes': ['human']}
    HEADERS = ['Current Step', 'Odds', 'Verbose Action', 'Action', 'Balance', 'Reward', 'Results', 'Done']

    def __init__(self, main_df, odds_column_names, num_possible_outcomes=3, results=None, starting_bank=1000):
        """Initializes a new environment

        Parameters
        ----------
        main_df: dataframe of shape (n_games, feats_dim)
            A list of games, with their the features the agent can access in training (including odds + additional info).
        odds_column_names: list of str
            A list of column names with length == n_odds.
        num_possible_outcomes: int
            Number of possible outcomes for a single game. i.e. 3 for soccer (home/draw/away), 2 for tennis / basketball (just home/away)
            Corresponds to the number of odds & bets to be (potentially) placed by the agent for each single match
        results: list of int, default=None
            A list of the results, where results[i] is the outcome of odds[i].
        """

        super().__init__()
        self._df = main_df.copy()
        self._odds_values_df = main_df.copy()[odds_column_names]
        self.ACTION_DIM = int(num_possible_outcomes)
        self.OBS_DIM = self._df.shape[1] # number of features the agent can access for each game (which includes odds + additional features)
        self._results = results
        self._odds_columns_names = odds_column_names
        self._verbose_actions = ActionsDict(self._odds_columns_names)
        self.observation_space = gym.spaces.Box(low=0.0, high=100.0, shape=(1, self.OBS_DIM), dtype=numpy.float64)
        self.action_space = gym.spaces.Discrete(2 ** self.ACTION_DIM)
        self.balance = starting_bank
        self.starting_bank = starting_bank
        self.current_step = 0
        self.bet_size_matrix = numpy.ones(shape=(self._df.shape[0], self.ACTION_DIM))

    def _get_current_index(self):
        return self.current_step % self._df.shape[0]

    def get_odds(self):
        """Returns the odds for the current step.

        Returns
        -------
        odds : numpy.ndarray of shape (1, n_odds)
            The odds for the current step.
        """
        return self._odds_values_df.iloc[self.current_step].values
    
    def get_obs(self):
        """Returns the feature-space observation for the current_step"""
        return self._df.iloc[self.current_step].values

    def get_bet(self, action):
        """Returns the betting matrix for the action provided.

        Parameters
        ----------
        action : int
            An action provided by the agent.

        Returns
        -------
        bet : array of shape (1, n_odds)
            The betting matrix, where each outcome specified in the action
            has a value of 1 and 0 otherwise.
        """
        return numpy.array([[bit for bit in numpy.binary_repr(action, width=self.ACTION_DIM)]]).astype(int)

    @pretty_traceback_and_exit_decorator
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of episode is reached,
        you are responsible for calling reset() to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters
        ----------
        action : int
            An action provided by the agent.

        Returns
        -------
        observation : dataframe of shape (1, OBS_DIM)
            The agent's observation of the current environment
        reward : float
            The amount of reward returned after previous action
        done : bool
            Whether the episode has ended, in which case further step() calls will return undefined results
        info : dict
            Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """
        odds = self.get_odds()
        reward = 0
        done = False
        info = self.create_info(action)
        if self.balance < 1:  # no more money :-(
            done = True
        else:
            bet = self.get_bet(action)
            results = self.get_results()
            if self.legal_bet(bet):  # making sure agent has enough money for the bet
                bet, odds, results = bet.reshape((-1, self.ACTION_DIM)), odds.reshape((-1, self.ACTION_DIM)), results.reshape((-1, self.ACTION_DIM))
                reward = self.get_reward(bet, odds, results)
                self.balance += reward
                info.update(legal_bet=True)
            else:
                reward = -(bet * self.bet_size_matrix).sum()
            info.update(results=results.argmax())
            info.update(reward=reward)
            self.current_step += 1
            if self.finish():
                done = True
                next = numpy.ones(shape=self.observation_space.shape)
            else:
                next = self.get_obs()
        info.update(done=done)
        return next, reward, done, info

    def get_reward(self, bet, odds, results):
        """ Calculates the reward

        Parameters
        ----------
        bet : array of shape (1, n_odds)
        odds: dataframe of shape (1, n_odds)
            A games with its betting odds.
        results : array of shape (1, n_odds)

        Returns
        -------
        reward : float
            The amount of reward returned after previous action
        """
        bet_size_matrix = self.bet_size_matrix  # noqa: F841
        returns = numexpr.evaluate('sum(bet * bet_size_matrix * results * odds)')
        expenses = numexpr.evaluate('sum(bet * bet_size_matrix)')
        profit = returns - expenses
        return profit

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns
        -------
        observation : dataframe of shape (1, n_odds)
            the initial observation.
        """
        self.balance = self.starting_bank
        self.current_step = 0
        return self._df.iloc[0].values
    

    def render(self, mode='human'):
        """Outputs the current balance and the current step.

        Returns
        -------
        msg : str
            A string with the current balance and the current step.
        """
        print('Current balance at step {}: {}'.format(self.current_step, self.balance))

    def finish(self):
        """Checks if the episode has reached an end.

        The episode has reached an end if there are no more games to bet.

        Returns
        -------
        finish : bool
            True if the current_step is equal to n_games, False otherwise
        """
        return self.current_step == self._df.shape[0]  # no more games left to bet

    def get_results(self):
        """Returns the results matrix for the current step.

        Returns
        -------
        result : array of shape (1, n_odds)
            The result matrix, where the index of the outcome that happened
            value is 1 and the rest of the indexes values are 0.
        """
        result = numpy.zeros(self.ACTION_DIM)
        label_index = self._results[self.current_step]
        result[label_index] = 1
        # result[numpy.arange(result.shape[0], dtype=numpy.int32),
        #        numpy.array([self._results[self.current_step]], dtype=numpy.int32)] = 1
        return result

    def legal_bet(self, bet):
        """Checks if the bet is legal.

        Checks that the bet does not exceed the current balance.

        Parameters
        ----------
        bet : array of shape (1, n_odds)
            The bet to check.

        Returns
        -------
        legal : bool
            True if the bet is legal, False otherwise.
        """
        return (bet * self.bet_size_matrix).sum() <= self.balance

    def create_info(self, action):
        """Creates the info dictionary for the given action.

        The info dictionary holds the following information:
            * the verbose action
            * the current step
            * the balance at the start of the current step
            * the relevant odds for the current step
            * the bet size for a single outcome

        Parameters
        ----------
        action : int
            An action provided by the agent.

        Returns
        -------
        info : dict
            The info dictionary.
        """
        return {'current_step': self.current_step, 'odds': self.get_odds(), 'verbose_action': self._verbose_actions[action],
                'action': str(action), 'balance': self.balance, 'reward': 0,
                'legal_bet': False, 'results': None, 'done': False}

    def pretty_print_info(self, info):
        values = [info[key.replace(' ', '_').lower()] for key in self.HEADERS]
        print('\n' + tabulate([values], headers=self.HEADERS, tablefmt='orgtbl'))

    def _rescale_form(self, form):
        if form == 1:
            form -= numpy.finfo(numpy.float64).eps
        return numpy.floor((form + 1) * (2 ** (self.ACTION_DIM - 1))).astype(int)

    def _rescale_matrix(self, bet_size_matrix):
        return numpy.abs(bet_size_matrix)
