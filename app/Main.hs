module Main where

import Lib

main :: IO ()
main = do
    net <- network [2,3,1]
    putStrLn "Network loaded again!"
