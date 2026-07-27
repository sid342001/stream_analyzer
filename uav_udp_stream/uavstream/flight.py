"""Synthetic UAV flight path: a sensor orbiting and staring at a fixed target."""

from __future__ import annotations

import math
import time

from .klv import UavTelemetry

EARTH_R = 6_378_137.0


def offset_latlon(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    """Flat-earth offset in metres - accurate enough for a few km."""
    dlat = math.degrees(north_m / EARTH_R)
    dlon = math.degrees(east_m / (EARTH_R * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


class OrbitPath:
    """Platform flies a circle around a ground target with the sensor staring at it."""

    def __init__(self, center_lat: float, center_lon: float, radius_m: float = 1500.0,
                 altitude_m: float = 1200.0, speed_mps: float = 45.0,
                 target_elev_m: float = 200.0, start_epoch: float | None = None) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius_m
        self.altitude = altitude_m
        self.speed = speed_mps
        self.target_elev = target_elev_m
        self.start_epoch = start_epoch if start_epoch is not None else time.time()
        self.omega = speed_mps / radius_m  # rad/s

    def at(self, t: float) -> UavTelemetry:
        """Telemetry at t seconds into the flight."""
        theta = self.omega * t
        north = self.radius * math.cos(theta)
        east = self.radius * math.sin(theta)
        lat, lon = offset_latlon(self.center_lat, self.center_lon, north, east)

        # velocity is tangent to the circle -> heading, plus a gentle bank
        heading = math.degrees(math.atan2(
            math.cos(theta) * self.radius * self.omega,
            -math.sin(theta) * self.radius * self.omega,
        )) % 360.0
        roll = -math.degrees(math.atan2(self.speed ** 2 / self.radius, 9.81))
        pitch = 2.0 * math.sin(theta * 0.5)

        # sensor stares at the orbit centre
        agl = self.altitude - self.target_elev
        ground = self.radius
        slant = math.hypot(agl, ground)
        depression = -math.degrees(math.atan2(agl, ground))
        bearing_to_center = (math.degrees(math.atan2(-east, -north))) % 360.0
        rel_az = (bearing_to_center - heading) % 360.0

        # a little wander so the frame centre is not pinned to one pixel
        jitter_n = 40.0 * math.sin(t * 0.7)
        jitter_e = 40.0 * math.cos(t * 0.5)
        fc_lat, fc_lon = offset_latlon(self.center_lat, self.center_lon, jitter_n, jitter_e)

        return UavTelemetry(
            timestamp_us=int((self.start_epoch + t) * 1_000_000),
            latitude=lat,
            longitude=lon,
            altitude=self.altitude + 15.0 * math.sin(t * 0.3),
            heading=heading,
            pitch=pitch,
            roll=roll,
            ground_speed=self.speed,
            rel_azimuth=rel_az,
            rel_elevation=depression,
            rel_roll=0.0,
            slant_range=slant,
            hfov=12.0 + 4.0 * math.sin(t * 0.15),
            vfov=7.0 + 2.3 * math.sin(t * 0.15),
            frame_center_lat=fc_lat,
            frame_center_lon=fc_lon,
            frame_center_elev=self.target_elev,
            target_width=250.0,
        )
