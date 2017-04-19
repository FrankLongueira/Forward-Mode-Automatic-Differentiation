-- Implementation of Forward Mode Automatic Differentiation in Haskell
-- By Frank Longueira

module AutoDiff where

-- "Dual" data type definition is just a 2-tuple of doubles.
-- Eq defines the methods (==) | (/=) entry-wise on two "Dual" ' s
-- Show & Read establishes I/O methods related to "Dual"
data Dual = Dual Double Double deriving (Eq, Show, Read)

-- In order to overload elementary operatiors like (+) and (*), we must
-- define an instance of the "Num" typeclass for our new data type "Dual." This requires
-- defining all methods related to this type class.
instance Num Dual where
	(Dual u ud) + (Dual v vd) = Dual (u + v) (ud + vd)
	(Dual u ud) * (Dual v vd) = Dual (u*v) (u*vd + v*ud)
	(Dual u ud) - (Dual v vd) = Dual (u - v) (ud - vd)
	abs(Dual u ud) = Dual (abs u) (ud * signum u)
	signum(Dual u ud) = Dual (signum u) 0
	fromInteger n = Dual (fromInteger n) 0
	
-- We provide an instance for the "Fractional" typeclass in order to take into account (/)
-- and rational numbers.
instance Fractional Dual where
  (Dual u ud) / (Dual v vd)   = Dual (u / v) (( v*ud - u*vd ) / v ** 2)
  recip (Dual u ud)           = Dual (recip u) (-ud * (recip (u ** 2)))
  fromRational n              = Dual (fromRational n) 0
  
  
-- We provide an instance for the "Floating" typeclass in order to compute derivatives involving
-- exponential, log, trig, hyperbolic, and other various functions.
instance Floating Dual where
  pi                = Dual pi 0
  exp (Dual u ud)   = Dual (exp u) (ud * exp u)
  log (Dual u ud)   = Dual (log u) (ud / u)
  sqrt (Dual u ud)  = Dual (sqrt u) (ud / (2 * sqrt u))
  (Dual u ud) ** (Dual n 0) = Dual (u**n) (ud*n*u**(n-1))
  (Dual a 0) ** (Dual v vd) = Dual (a**v) (vd*log(a)*(a**v))
  (Dual u ud) ** (Dual v vd) = Dual (u ** v) (u ** v * (vd * (log u) + (v * ud / u)))
  logBase (Dual u ud) (Dual v vd) = Dual (logBase u v) (((log v) * ud / u - (log u) * vd / v) / ((log u) ** 2))
  sin (Dual u ud)   = Dual (sin u) (ud * cos u)
  cos (Dual u ud)   = Dual (cos u) (- ud * sin u)
  tan (Dual u ud)   = Dual (tan u) (1 / ((cos u) ** 2))
  asin (Dual u ud)  = Dual (asin u) (ud / (sqrt(1 - u ** 2)))
  acos (Dual u ud)  = Dual (acos u) (- ud / (sqrt(1 - u ** 2)))
  atan (Dual u ud)  = Dual (atan u) (ud / (1 + u ** 2))
  sinh (Dual u ud)  = Dual (sinh u) (ud * cosh u)
  cosh (Dual u ud)  = Dual (cosh u) (ud * sinh u)
  tanh (Dual u ud)  = Dual (tanh u) (ud * (1 - (tanh u) ** 2))
  asinh (Dual u ud) = Dual (asinh u) (ud / (sqrt(1 + u ** 2)))
  acosh (Dual u ud) = Dual (acosh u) (ud / (sqrt(u ** 2 - 1)))
  atanh (Dual u ud) = Dual (atanh u) (ud / (1 - u ** 2))

-- This function extracts the function value (first) component of a dual number.
getValue :: (Dual) -> Double
getValue (Dual x _) = x

-- This function extracts the derivative (second) component of a dual number.
getDeriv :: (Dual) -> Double
getDeriv (Dual _ xd) = xd

-- This function seeds a dual number with a 1 in its derivative component
seed :: Double -> Dual
seed x = Dual x 1

-- This function takes as input a function f:Dual -> Dual & x-value and returns
-- the derivative value at that x-value.
diff :: (Dual -> Dual) -> Double -> Double
diff f x =  getDeriv $ f (Dual x 1)

-- Compute derivative of function over a list of given x-values
diff_many :: (Dual -> Dual) -> [Double] -> [Double]
diff_many f x = map (diff f) $ x

-- Compute gradient of two-variable scalar function
diff2 :: (Dual -> Dual -> Dual) ->  Double -> Double -> (Double, Double)
diff2 f x y = ( getDeriv $ f (Dual x 1) (Dual y 0) , getDeriv $ f (Dual x 0) (Dual y 1) )

-- Compute gradient of three-variable scalar function
diff3 :: (Dual -> Dual -> Dual -> Dual) -> Double -> Double -> Double -> (Double, Double, Double)
diff3 f x y z = ( getDeriv $ f (Dual x 1) (Dual y 0) (Dual z 0), getDeriv $ f (Dual x 0) (Dual y 1) (Dual z 0), getDeriv $ f (Dual x 0) (Dual y 0) (Dual z 1))


-- Algorithm 1 - Function composition N times (recursively implemented)
-- AD allows for efficient computation & avoids expression swell if done algebraically.
f_comp :: (Dual->Dual) -> Integer -> Dual -> Dual
f_comp f 1 x = f x
f_comp f n x = f_comp f (n-1) ( f x )


-- Algorithm 2 - Control Flow Statement Algorithm
-- AD allows for derivative computation of control flow statements, but the user must be aware if the output makes sense. (i.e. at non-differentiable points)
control_flow_algo :: Dual -> Dual
control_flow_algo x
	| (getValue x) < 0 = exp( x )
	| (getValue x) >= 0 = cos( x )


