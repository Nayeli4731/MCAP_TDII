import logging
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path
from scipy.stats import levy_stable, wrapcauchy
from scipy.spatial import distance
from typing import Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Vec2d(object):
    """
    2d vector class, supports vector and scalar operators, and also provides a bunch of high level functions
    http://www.pygame.org/wiki/2DVectorClass
    """
    __slots__ = ['x', 'y']

    def __init__(self, x_or_pair, y=None):
        if y == None:
            self.x = x_or_pair[0]
            self.y = x_or_pair[1]
        else:
            self.x = x_or_pair
            self.y = y

    # Addition
    def __add__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x + other.x, self.y + other.y)
        elif hasattr(other, "__getitem__"):
            return Vec2d(self.x + other[0], self.y + other[1])
        else:
            return Vec2d(self.x + other, self.y + other)

    # Subtraction
    def __sub__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x - other.x, self.y - other.y)
        elif (hasattr(other, "__getitem__")):
            return Vec2d(self.x - other[0], self.y - other[1])
        else:
            return Vec2d(self.x - other, self.y - other)

    # Vector length
    def get_length(self):
        return math.sqrt(self.x**2 + self.y**2)

    # rotate vector
    def rotated(self, angle):
        cos = math.cos(angle)
        sin = math.sin(angle)
        x = self.x*cos - self.y*sin
        y = self.x*sin + self.y*cos
        return Vec2d(x, y)


class RandomWalk:
    """
    Class that contains all the methods related to RandomWalk
    """
    def path_length_calculation(trajectory: pd.DataFrame) -> np.array:
        """_summary_

        :param trajectory: _description_
        :type trajectory: pd.DataFrame
        :return: _description_
        :rtype: np.array
        """
        eu = 0
        if isinstance(trajectory, pd.DataFrame):
            eu = euclidean_distance(trajectory)
        else:
            logger.error(f"Please insert DataFrame type")

        return eu

    def euclidean_distance(df: pd.DataFrame) -> np.array:
        """_summary_

        :param df: _description_
        :type df: pd.DataFrame
        :return: _description_
        :rtype: np.array
        """
        if isinstance(trajectory, pd.DataFrame):
            return np.array([math.sqrt(math.pow((df.x_pos[i+1] - df.x_pos[i]), 2) +
                                       math.pow((df.y_pos[i+1] - df.y_pos[i]), 2))
                             for i in range(df.shape[0]-1)])
        else:
            logger.error(f"Please insert DataFrame type")
            return None

    def single_euclidean_distance(pointA: pd.core.series.Series, pointB: pd.core.series.Series) -> float:
        """_summary_

        :param pointA: _description_
        :type pointA: pd.core.series.Series
        :param pointB: _description_
        :type pointB: pd.core.series.Series
        :return: _description_
        :rtype: float
        """
        ed = math.sqrt(math.pow((pointB[0] - pointA[0]), 2) + math.pow((pointB[1] - pointA[1]), 2))
        return ed

    def plot_figure(traces: list, plot_type: str, dimension: str = "2d"):
        """_summary_

        :param traces: _description_
        :type traces: list
        :param plot_type: _description_
        :type plot_type: str
        :param dimension: _description_, defaults to "2d"
        :type dimension: str, optional
        :return: _description_
        :rtype: _type_
        """
        figure = go.Figure()

        if dimension == "2d":
            if plot_type == "lineal":
                for trace in traces:
                    figure.add_trace(go.Scatter(
                        x=trace["x"],
                        y=trace["y"] if "y" in trace.keys() else None,
                        marker=dict(size=2),
                        line=dict(width=2),
                        mode="lines",
                        name=f"{trace['name']}",
                        showlegend=True
                    ))
            elif plot_type == "histogram":
                for trace in traces:
                    figure.add_trace(go.Histogram(
                        x=trace["x"],
                        nbinsx=trace["nbinsx"] if "nbinsx" in trace.keys() else None,
                        xbins=dict(start=trace["xbins"]["start"], end=trace["xbins"]
                                   ["end"], size=trace["xbins"]["size"]) if "xbins" in trace.keys() else None,
                        name=f"{trace['name']}",
                        opacity=trace["opacity"] if "opacity" in trace.keys() else None,
                        histnorm="probability density",
                        marker_color=trace["color"] if "color" in trace.keys() else None,
                        showlegend=True))
                figure.update_layout(bargap=0.4)

            elif plot_type == "mix":
                for trace in traces:
                    if "y" in trace.keys():
                        figure.add_trace(go.Scatter(
                            x=trace["x"],
                            y=trace["y"],
                            marker=dict(size=2),
                            line=dict(width=2),
                            mode="lines",
                            name=f"{trace['name']}",
                            showlegend=True
                        ))
                    else:
                        figure.add_trace(go.Histogram(
                            x=trace["x"],
                            nbinsx=trace["nbinsx"] if "nbinsx" in trace.keys() else None,
                            xbins=dict(start=trace["xbins"]["start"], end=trace["xbins"]
                                       ["end"], size=trace["xbins"]["size"]) if "xbins" in trace.keys() else None,
                            name=f"{trace['name']}",
                            opacity=trace["opacity"] if "opacity" in trace.keys() else None,
                            histnorm="probability density",
                            marker_color=trace["color"] if "color" in trace.keys() else None,
                            showlegend=True))

                figure.update_layout(bargap=0.4)
            else:
                logger.error(f"For 2D dimensions insert a valid type to be plotted")
                return False
        elif dimension == "3d":
            if plot_type == "lineal":
                for trace in traces:
                    figure.add_trace(go.Scatter3d(x=trace["x"],
                                                  y=trace["y"],
                                                  z=trace["z"],
                                                  marker=dict(size=2),
                                                  line=dict(
                                                      color=f"{trace['color']}" if "color" in trace.keys() else "blue", width=2),
                                                  mode="lines",
                                                  name=f"{trace['name']}",
                                                  showlegend=True))
            else:
                logger.error(f"For 3D dimensions only lineal functions can be plotted")
                return False

        figure.show()

    def calculate_vector(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> list:
        """_summary_

        :param a: _description_
        :type a: pd.DataFrame
        :param b: _description_
        :type b: pd.DataFrame
        :param c: _description_
        :type c: pd.DataFrame
        :return: _description_
        :rtype: list
        """
        ab = [b.x_pos-a.x_pos, b.y_pos-a.y_pos]
        bc = [c.x_pos-b.x_pos, c.y_pos-b.y_pos]
        return ab, bc

    def angle_between_arcos(v1: list, v2: list):
        """_summary_

        :param v1: _description_
        :type v1: list
        :param v2: _description_
        :type v2: list
        :return: _description_
        :rtype: _type_
        """
        return np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))

    def angle_between_arcsin(v1: list, v2: list):
        """_summary_

        :param v1: _description_
        :type v1: list
        :param v2: _description_
        :type v2: list
        :return: _description_
        :rtype: _type_
        """
        return np.arcsin(np.cross(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))

    def read_csv(file_path: str) -> pd.DataFrame:
        """_summary_

        :param file_path: _description_
        :type file_path: str
        :return: _description_
        :rtype: pd.DataFrame
        """
        if Path(file_path).exists():
            return pd.read_csv(file_path)
        else:
            logger.error(f"File does not exists, please review")
            return False

    def generate_distribution_function(resolution: int, c: float = None, alpha: float = None,
                                       beta: float = None, loc: float = None, distribution_type: str = "couchy") -> np.array:
        """_summary_

        :param resolution: _description_
        :type resolution: int
        :param c: _description_, defaults to None
        :type c: float, optional
        :param alpha: _description_, defaults to None
        :type alpha: float, optional
        :param beta: _description_, defaults to None
        :type beta: float, optional
        :param loc: _description_, defaults to None
        :type loc: float, optional
        :param distribution_type: _description_, defaults to "couchy"
        :type distribution_type: str, optional
        :return: _description_
        :rtype: np.array
        """
        if distribution_type == "couchy":
            aux_domain = np.linspace(0, 2*np.pi, resolution)
            distribution_pdf = np.array([wrapcauchy.pdf(i, c) for i in aux_domain])
        else:
            aux_domain = np.linspace(0, 50, resolution)
            distribution_pdf = np.array([levy_stable.pdf(i, alpha, beta, loc) for i in aux_domain])

        return distribution_pdf

    def generate_csv(data: pd.DataFrame, name: str):
        """_summary_

        :param data: _description_
        :type data: pd.DataFrame
        :param name: _description_
        :type name: str
        """
        data.to_csv(name=f"{name}.csv", encoding="utf-8")

    def calculate_turning_angle(dataframe: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        :param dataframe: _description_
        :type dataframe: pd.DataFrame
        :return: _description_
        :rtype: pd.DataFrame
        """
        N = dataframe.shape[0]
        aux_ta = np.zeros([N-2])
        for i in range(1, N-1):
            AB, BC = calculate_vector(dataframe.iloc[i-1], dataframe.iloc[i], dataframe.iloc[i+1])
            if np.cross(AB, BC) > 0:
                angle = angle_between_arcos(AB, BC)
                aux_ta[i-1] = 0 if math.isnan(angle) else angle
            else:
                angle = angle_between_arcos(AB, BC)
                aux_ta[i-1] = 0 if math.isnan(angle) else -angle

        metrics_ta_df = pd.DataFrame(columns=["TA_CRW"])
        for ta_crw in aux_ta:
            temp_ta_df = pd.DataFrame([{"TA_CRW": ta_crw}])
            metrics_ta_df = pd.concat([metrics_ta_df, temp_ta_df], ignore_index=True)

        return metrics_ta_df

    def calculate_step_length_distribution(dataframe: pd.DataFrame):
        """_summary_

        :param dataframe: _description_
        :type dataframe: pd.DataFrame
        :return: _description_
        :rtype: _type_
        """
        N = dataframe.shape[0]
        aux_sl = np.zeros([N-2])
        steps = 1
        data_sl_df = pd.DataFrame(columns=["TA_value", "Steps"])
        for idx in range(len(aux_sl)-1):
            if math.isclose(aux_sl[idx], aux_sl[idx+1], rel_tol=1e-2) or round(aux_sl[idx], ndigits=2) == 0:
                steps += 1
            else:
                temp_df = pd.DataFrame([{"TA_value": aux_sl[idx], "Steps": steps}])
                data_sl_df = pd.concat([data_sl_df, temp_df], ignore_index=True)
                steps = 1
        return data_sl_df

    def calculate_mean_sqrt_displacement(dataframe: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        :param dataframe: _description_
        :type dataframe: pd.DataFrame
        :return: _description_
        :rtype: pd.DataFrame
        """
        # Empty MSD
        N = dataframe.shape[0]
        MSD = np.zeros([N])
        cum_sum = 0

        for tau in range(1, N):
            for i in range(1, N-tau+1):
                cum_sum += math.pow(single_euclidean_distance(dataframe.iloc[i+tau-1], dataframe.iloc[i-1]), 2)
            MSD_CRW[tau] = (cum_sum/(N-tau))
            cum_sum = 0

        # Save metrics to Dataframe
        metrics_df = pd.DataFrame(columns=["MSD"])
        for metric in MSD:
            temp_df = pd.DataFrame([{"MSD": metric}])
            metrics_df = pd.concat([metrics_df, temp_df], ignore_index=True)

        return metrics_df

    def brownian_motion(n_steps: int = 1000, speed: int = 6, s_pos: list = [0, 0]) -> pd.DataFrame:
        """_summary_

        :param n_steps: _description_, defaults to 1000
        :type n_steps: int, optional
        :param speed: _description_, defaults to 6
        :type speed: int, optional
        :param s_pos: _description_, defaults to [0, 0]
        :type s_pos: list, optional
        :return: _description_
        :rtype: pd.DataFrame
        """
        # Init velocity
        velocity = Vec2d(x_or_pair=speed, y=0)
        # Init dataframe
        BM_2d_df = pd.DataFrame(columns=["x_pos", "y_pos"])
        temp_df = pd.DataFrame([{"x_pos": s_pos[0], "y_pos": s_pos[1]}])

        BM_2d_df = pd.concat([BM_2d_df, temp_df], ignore_index=True)

        for i in range(n_steps-1):
            turn_angle = np.random.uniform(low=-np.pi, high=np.pi)  # TYPE OF DISTRIBUTION
            velocity = velocity.rotated(turn_angle)
            temp_df = pd.DataFrame([{"x_pos": BM_2d_df.x_pos[i]+velocity.x, "y_pos":  BM_2d_df.y_pos[i]+velocity.y}])
            BM_2d_df = pd.concat([BM_2d_df, temp_df], ignore_index=True)

        return BM_2d_df

    def correlated_random_walk(crw_exponent, n_steps: int = 1000, s_pos: list = [0, 0], speed: int = 6) -> pd.DataFrame:
        """_summary_

        :param crw_exponent: _description_
        :type crw_exponent: _type_
        :param n_steps: _description_, defaults to 1000
        :type n_steps: int, optional
        :param s_pos: _description_, defaults to [0, 0]
        :type s_pos: list, optional
        :param speed: _description_, defaults to 6
        :type speed: int, optional
        :return: _description_
        :rtype: pd.DataFrame
        """
        # Init velocity
        velocity = Vec2d(x_or_pair=speed, y=0)

        # Init dataframe
        CRW_2d_df = pd.DataFrame(columns=["x_pos", "y_pos"])

        # Concatenate aux
        temp_df = pd.DataFrame([{"x_pos": s_pos[0], "y_pos": s_pos[1]}])
        CRW_2d_df = pd.concat([CRW_2d_df, temp_df], ignore_index=True)

        for i in range(n_steps - 1):
            turn_angle = wrapcauchy.rvs(c=CRW_exponent)
            velocity = velocity.rotated(turn_angle)
            temp_df = pd.DataFrame([{"x_pos": CRW_2d_df.x_pos[i]+velocity.x, "y_pos":  CRW_2d_df.y_pos[i]+velocity.y}])
            CRW_2d_df = pd.concat([CRW_2d_df, temp_df], ignore_index=True)

        return CRW_2d_df

    def levy_flight(alpha: float = 1.0, beta: float = 1.0, std_motion: int = 6, speed: int = 1, n_samples: int = 100000,
                    s_pos: list = [0, 0], c: float = 0.7, loc: float = 0) -> pd.DataFrame:
        """_summary_

        :param alpha: _description_, defaults to 1.0
        :type alpha: float, optional
        :param beta: _description_, defaults to 1.0
        :type beta: float, optional
        :param std_motion: _description_, defaults to 6
        :type std_motion: int, optional
        :param speed: _description_, defaults to 1
        :type speed: int, optional
        :param n_samples: _description_, defaults to 100000
        :type n_samples: int, optional
        :param s_pos: _description_, defaults to [0, 0]
        :type s_pos: list, optional
        :param c: _description_, defaults to 0.7
        :type c: float, optional
        :param loc: _description_, defaults to 0
        :type loc: float, optional
        :return: _description_
        :rtype: pd.DataFrame
        """

        # init velocity vector
        velocity = Vec2d(x_or_pair=speed, y=0)  # inicia horizontal

        # init DF
        LW_2d_df = pd.DataFrame([{"x_pos": s_pos[0], "y_pos": s_pos[1]}])

        i = 1
        while i < n_samples:
            # Get random n_steps from levy
            step_size = levy_stable.rvs(alpha, beta, std_motion)
            # round to int and positive
            step_size = int(np.ceil(abs(step_size)))
            # angle
            theta = wrapcauchy.rvs(c=c, loc=loc)
            velocity = velocity.rotated(theta)
            for j in range(step_size):
                temp_df = pd.DataFrame([{"x_pos": LW_2d_df.x_pos[i-1] + velocity.x,
                                        "y_pos":  LW_2d_df.y_pos[i-1] + velocity.y}])
                # add to the end to levy df
                LW_2d_df = pd.concat([LW_2d_df, temp_df], ignore_index=True)
                i += 1
        return LW_2d_df
