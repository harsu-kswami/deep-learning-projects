class EMASmoother:
    """
    Simple exponential moving average smoother for steering.
    smooth = prev + alpha * (new - prev)
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None

    def reset(self):
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.value + self.alpha * (x - self.value)
        return self.value


class PIDController:
    """
    Generic PID controller. You can use this
    on top of model steering to smooth + stabilize.
    """
    def __init__(self, kp=0.5, ki=0.0, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def step(self, target, current, dt=0.05):
        """
        target: desired steering
        current: current (previous) steering command
        dt: time step (s)
        """
        error = target - current
        self.integral += error * dt

        derivative = 0.0
        if self.prev_error is not None:
            derivative = (error - self.prev_error) / dt

        self.prev_error = error

        out = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )
        # final steering command
        return current + out
