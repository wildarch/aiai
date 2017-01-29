module Lib
    ( Network,
      network
    ) where

import Numeric.LinearAlgebra


data Network = Network {
  biases :: [Matrix R],
  weights :: [Matrix R]
}

network :: [Int] -> IO Network
network sizes = do
  biases <- sequence $ map (randn 1) $ tail sizes
  weights <- sequence $ map (\(a,b) -> randn b a) $ zip (init sizes) (tail sizes)
  return $ Network biases weights
