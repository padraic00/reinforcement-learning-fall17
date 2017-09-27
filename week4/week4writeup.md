# Reinforcement Learning in Trading
##What is Reinforcement Learning?
Machine Learning is a multi-faceted field that is best described by the different method it offers in solving numerical problems. The two main branches of machine learning are supervised and unsupervised learning. Supervised learning involves the author of the algorithm telling the user what a positive and negative output is, and points to what methods to use to solve the problem. Unsupervised learning involves the author completely defining the goal of the algorithm and the rewards for every action the agent takes, but having the agent choose its own actions based on its own estimate of the rewards. Unlike a regression, or any other supervised learning method, the agent can create a full solution to a problem, which becomes useful in the world of automated trading.

## Why is Reinforcement Learning useful for Trading?

Many different Machine Learning algorithms can be used to quantify the changes in asset prices from different input variables. Typically, one may use the simplest form of Supervised Learning: a regression. A regression, specifically a linear one, can be used to find coefficients that can predict the next asset return based on the previous one as well as multiple different indicators. However, the issue with supervised learning algorithms is that while they may be engineered to be accurate, they do not generate a trading *policy*. A regression does not tell a trader when to add long and short positions, how to manage risk, how to react to massive market movements. Instead, the output of a converged Reinforcement Learning Algorithm is a policy that maximizes the value for every state by taking an associated action. The learner can internalize concepts like risk management, position concentration, and most importantly adapts to the current pattern of market trends: it learns how a time series moves from one state to another. 

## Past Applications:
The most commonly used algorithm from the world of reinforcement learning in creating an automated trader is **Q-Learning**. Q-Learning is a form of  Temporal Difference Reinforcement Learning that can modify its internal policy to reflect what it believes is the optimal policy at any time step, and uses the following equation to compute the update:
![enter image description here](https://lh3.googleusercontent.com/-vgLhy3cI38E/Wcr22mLLehI/AAAAAAAAKcQ/bJ5tyqA0UNEx-2P9I4la7pkhxFWyCiwfwCLcBGAs/s0/Screen+Shot+2017-09-26+at+8.27.30+PM.png "Q Function")

The simulated PnL from an implementation of Q-Learning from simulated market movements of a stochastic process look like this:
![enter image description here](https://lh3.googleusercontent.com/-Q2Qp-vuKNPI/Wcr3wyTHdcI/AAAAAAAAKck/A0Uemf94-UgBw5BRnbzGL0akgKltCx37ACLcBGAs/s0/Screen+Shot+2017-09-26+at+8.56.44+PM.png "Screen Shot 2017-09-26 at 8.56.44 PM.png") [1]

## Market Choice
The results from Ritter's paper are interesting, but the data set he chooses to apply it on is simulation generated: it is generated from a simply parametrized stochastic process, and does not include any distribution tail or Black Swan events. Our project should be centered around apply this method of Reinforcement Learning to an existing and exciting market. We have made the choice to apply this method to a rapidly evolving investment space: cryptocurrencies. First, a few definitions:

 - Blockchain is the concept of distributing centralized records of information to a large number of decentralized ledgers. These ledgers agree on what transactions are legal or not legal using a voting system. The specifics of how each system operates is up to the creator, and the investment space is based around observers and developers betting money on which technology will become the mainstream information system of the future. The primary application of blockchain technology is in financial record keeping, with the ledgers being encrypted transaction records from one wallet to another.
 - Cryptocurrencies are the underlying token that allow blockchain environments or applications to exist. Transactions existing on the Bitcoin network are conducted using Bitcoin, those transacted on the Ether chain are conducted using Ethereum.
 - Cryptocurrency markets are publicly accessible electronic exchanges that allow users to exchange fiat currency for cryptocurrencies, like Bitcoin and Ethereum, or cryptocurrency for cryptocurrency, with the investment hypothesis being that certain cryptocurrencies will outperform others based on their popularity, acceptance and ability to perform as the next worldwide financial transaction system. The inherent value is obvious: for one to use this new transaction system, one must have the cryptocurrency in hand. If the demand for this cryptocurrency goes up, the value will increase due to scarcity. 

### Why Crypto?
Cryptocurrencies, in comparison to nearly every other exchange traded instrument, has the least amount of institutional investment involved: most investors are tech professionals that hold the currencies for long periods of time because they believe in the underlying technology succeeding the current financial system. This compares well with the retail "value" investor, who believes in the fundamental product that a company offers and for this reason invests in the company stock. Some investors in the cryptocurrency space use the currency for transactions: indeed, businesses are starting to accept the currency for products just as Canadian businesses are capable of accepting US Dollars in electronic transactions by checking the exchange rate at the time of purchase. 

Varied investment thesis, as well as active trading flow moving in and out of the markets, make for an ideal starting ground for the project: because institutions with proper investment professionals have not entered the space with proper trading strategies it should be easier to train an agent on these products. In addition, it is easier to access high resolution, uncleaned market data than in more evolved products: we have already collected minute resolution market data for 3 major currencies traded on GDAX (the market exchange for Coinbase) for 6 months, a feat impossible for free in US equities. 

## Project Goal:
We will segment the goals into a minimum and reach goals, with the minimum goals being the ones that would sufficiently qualify us for passing the independent study, and the reach goals being the ones that we hope but cannot guarantee to be successful. 
### Minimum Goals:
 - Finish theoretical material from Sutton's *Reinforcement Learning*
 - Replicate performance from Ritter's *Machine Learning for Trading* on crypto currency markets
 - Walk forward testing and performance tracking of the trader with unseen market conditions.
### Reach Goals:
 - Increase performance with modifications to algorithm
	 - Improve convergence time, backtest statistics
 - Use understanding of states from the Unsupervised Reinforcement Learner to create a SVM classifier of market movements

## Next Steps

 - What determines a state? Previous research indicates the use of two main indicators
	 - Momentum
	 - Mean Reversion
	 - Hypothesis that the space of market movements is defined by these two mathematical concepts
 - Previous research also indicates the inclusion of previous steps return and current positioning as a state definition as well
	 - Allows the agent to internalize the concepts of risk, portfolio management, take profit and stop loss
	 - These parameters can change based on the previous performance (ie take profit and stop loss levels change when you have made 1% return versus 20% return)
 - How do we reward the agent?
	 - Do we give it a +1 reward when it meets a certain annual returns figure?
	 - Do we reward for positive PnL every day and discount for negative PnL?
 - Technical Aspects of Implementation
	 - Dynamically allocating state function space because of episodic state definitions from return and position
	 - Length of data per training episode
	 - How many training episodes
	 - Out of Sample

### Bibliography:

 1.  Ritter, Gordon. “Machine Learning for Trading.” SSRN, New York University, 14 Aug. 2017. 