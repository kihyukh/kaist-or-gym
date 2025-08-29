import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Car:
    def __init__(self, position):
        self.position = position
        self.is_moving = True
        self.is_done = False
        self.total_wait_time = 0.0
        self.total_move_time = 0.0


class TrafficControlEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 30,
    }

    def __init__(self, *, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.arrival_rates = {"N": 6 / 60, "E": 3 / 60, "S": 3 / 60, "W": 3 / 60}
        self.car_velocity = 30 * 1000 / (60 * 60)
        self.road_length = 200
        self.road_width = 20
        self.stop_distance = 5
        self.min_distance = 25  # Added minimum distance
        self.t = 0.0
        self.dt = 0.1
        self.signal = "RR"
        self._time_in_signal_state = 0.0
        self._yellow_light_duration = 2.2 * (self.road_width / self.car_velocity)
        self._target_signal = "RR"


        self.active_cars = {
            "N": [],
            "S": [],
            "E": [],
            "W": [],
        }
        self.finished_cars = []  # Add this line to track finished cars

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(), dtype=float)

        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            try:
                from IPython.display import display
            except ImportError:
                display = None
            self._plt = plt
            self._display = display

            self.fig, self.ax = None, None
            self.display_handle = None

    def _get_observation(self):
        # TODO: Implement a meaningful observation such as number of cars waiting in each direction.
        return np.array(0)

    def _get_info(self):
        return {
            "target_signal": self._target_signal,
        }

    def _is_green(self, direction):
        if direction in ("N", "S") and self.signal[0] == "G":
            return True
        if direction in ("E", "W") and self.signal[1] == "G":
            return True
        # Return False if the signal for the given direction is yellow
        if direction in ("N", "S") and self.signal[0] == "Y":
            return False
        if direction in ("E", "W") and self.signal[1] == "Y":
            return False
        return False


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0.0
        self.signal = "RR"
        self._time_in_signal_state = 0.0
        self._target_signal = "RR"
        self.active_cars = {
            "N": [],
            "S": [],
            "E": [],
            "W": [],
        }
        self.finished_cars = []  # Reset finished cars on env reset
        if self.render_mode == "human":
            if self.fig is not None:
                self._plt.close(self.fig)
            if self._plt is None or self._display is None:
                import matplotlib.pyplot as plt
                try:
                    from IPython.display import display
                except ImportError:
                    display = None
                self._plt = plt
                self._display = display
            self.fig, self.ax = self._plt.subplots(figsize=(5,5))
            self.display_handle = self._display(self.fig, display_id=True) if self._display is not None else None
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.t += self.dt
        self._time_in_signal_state += self.dt

        # Implement yellow light logic
        if self.signal == "GR" and action == 2:
            self._target_signal = "RG"
            self.signal = "YR"
            self._time_in_signal_state = 0.0
        elif self.signal == "RG" and action == 1:
            self._target_signal =  "GR"
            self.signal = "RY"
            self._time_in_signal_state = 0.0
        elif self.signal in ("YR", "RY") and self._time_in_signal_state >= self._yellow_light_duration:
            self.signal = self._target_signal
            self._time_in_signal_state = 0.0
        elif self.signal in ("YR", "RY"):
            # Ignore new actions during yellow light
            pass
        else:
            # Change signal based on action for red lights
            if action == 1:
                self.signal = "GR" # Green for North/South, Red for East/West
                self._target_signal = "GR" # Green for North/South, Red for East/West
            elif action == 2:
                self.signal = "RG" # Red for North/South, Green for East/West
                self._target_signal = "RG" # Red for North/South, Green for East/West


        # generate cars
        for direction in ["N", "S", "E", "W"]:
            if np.random.random() > self.dt * self.arrival_rates[direction]:
                continue
            self.active_cars[direction].append(Car(-self.road_length-self.road_width))

        # move cars
        for direction, car_list in self.active_cars.items():
            num_cars_to_remove = 0
            for i, car in enumerate(car_list):
                # Calculate the position of the car in front, if any
                leading_car_position = self.road_length + self.road_width + self.min_distance # Default to far away
                if i > 0:
                    leading_car = car_list[i-1]
                    leading_car_position = leading_car.position

                # Determine the maximum allowable position for the current car
                max_position = leading_car_position - self.min_distance

                next_car_position = car.position

                if self._is_green(direction) or car.position > -self.road_width:
                    # green light or the car started crossing or already passed the crossroad
                    next_car_position += self.car_velocity * self.dt
                    car.total_move_time += self.dt
                else: # yellow or red light
                    stop_limit = min(car_list[i-1].position, -self.road_width ) \
                            if i > 0 else -self.road_width
                    stop_limit -= self.stop_distance

                    if car.position + self.car_velocity * self.dt < stop_limit:
                        next_car_position += self.car_velocity * self.dt
                        car.total_move_time += self.dt
                    else:
                        next_car_position = max(car.position, stop_limit)
                        car.total_wait_time += self.dt

                # Enforce minimum distance, unless the leading car is stopped at a red light
                if i > 0 and (self._is_green(direction) or leading_car.is_moving):
                     next_car_position = min(next_car_position, max_position)

                # Ensure the car"s position does not decrease
                next_car_position = max(car.position, next_car_position)

                # Update is_moving based on whether the position changed
                car.is_moving = car.position < next_car_position
                car.position = next_car_position


                if car.position > self.road_length + self.road_width:
                    car.is_done = True
                    num_cars_to_remove += 1
            # Remove finished cars and store them in finished_cars
            finished = [car for car in car_list if car.is_done]
            self.finished_cars.extend(finished)
            self.active_cars[direction] = [car for car in car_list if not car.is_done]


        observation = self._get_observation()
        info = self._get_info()
        # In this simple environment, there is no reward, terminated, or truncated condition.
        reward = 0.0
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated, info


    def get_avg_waiting_time(self):
        all_cars = []
        for cars in self.active_cars.values():
            all_cars.extend(cars)
        all_cars.extend(self.finished_cars)
        total_cars = len(all_cars)
        total_wait_time = sum(car.total_wait_time for car in all_cars)
        avg_waiting_time = total_wait_time / total_cars if total_cars > 0 else 0.0
        return avg_waiting_time


    def render(self):
        if self.render_mode != "human":
            return
        # Ensure plt and display are available
        if self._plt is None or self._display is None:
            import matplotlib.pyplot as plt
            try:
                from IPython.display import display
            except ImportError:
                display = None
            self._plt = plt
            self._display = display
        # Ensure fig and ax are available
        if self.fig is None or self.ax is None:
            self.fig, self.ax = self._plt.subplots(figsize=(5,5))

        self.ax.cla() # Clear the previous plot
        self.ax.set_title( r"$t=%.1f$s"%self.t )

        dirs = ["N", "S", "E", "W"]
        ncars = {d:len(self.active_cars[d]) for d in dirs}
        self.ax.plot( -0.5*self.road_width*np.ones(ncars["N"]), [-car.position for car in self.active_cars["N"]], "v", c="b", alpha=0.5 )
        self.ax.plot( 0.5*self.road_width*np.ones(ncars["S"]), [car.position for car in self.active_cars["S"]], "^", c="b", alpha=0.5 )
        self.ax.plot( [-car.position for car in self.active_cars["E"]], 0.5*self.road_width*np.ones(ncars["E"]), "<", c="b", alpha=0.5 )
        self.ax.plot( [car.position for car in self.active_cars["W"]], -0.5*self.road_width*np.ones(ncars["W"]), ">", c="b", alpha=0.5 )

        self.ax.axvline( -self.road_width, c="k", lw=1 )
        self.ax.axvline( 0, c="y", lw=1, ls="--" )
        self.ax.axvline( self.road_width, c="k", lw=1 )
        self.ax.axhline( -self.road_width, c="k", lw=1 )
        self.ax.axhline( 0, c="y", lw=1, ls="--" )
        self.ax.axhline( self.road_width, c="k", lw=1 )

        ns_color = self.signal[0].lower()
        ew_color = self.signal[1].lower()

        self.ax.plot( [-0.5*self.road_width, 0.5*self.road_width], [self.road_width, -self.road_width], "o", c=ns_color )
        self.ax.plot( [-self.road_width, self.road_width], [-0.5*self.road_width, 0.5*self.road_width], "o", c=ew_color )


        self.ax.set_xlim( -self.road_width-self.road_length, self.road_width+self.road_length )
        self.ax.set_ylim( -self.road_width-self.road_length, self.road_width+self.road_length )

        # Calculate average waiting time for all cars (active + finished)
        all_cars = []
        for cars in self.active_cars.values():
            all_cars.extend(cars)
        all_cars.extend(self.finished_cars)
        total_cars = len(all_cars)
        total_wait_time = sum(car.total_wait_time for car in all_cars)
        avg_waiting_time = total_wait_time / total_cars if total_cars > 0 else 0.0

        description = " # of cars=%d\n" % sum(ncars.values())
        description += " # of waiting cars=%d\n" % sum(not car.is_moving for direction_list in self.active_cars.values() for car in direction_list)
        description += " avg waiting=%.2fs"% avg_waiting_time
        self.ax.text( self.ax.get_xlim()[0], self.ax.get_ylim()[1], description, va="top", ha="left" )

        # Update the displayed plot
        if self.display_handle is not None:
            self.display_handle.update(self.fig)
        else:
            self._plt.pause(0.001)
