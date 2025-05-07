#!/usr/bin/env python3
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor


class Telemetry:
    """A simplified interface for OpenTelemetry instrumentation"""

    def __init__(
        self,
        service_name: str,
        service_version: str = "0.1.0",
        environment: str = "production",
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.enabled = all(
            os.getenv(k)
            for k in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_HEADERS")
        )

        self.tracer = None
        self.meter = None
        self._counters = {}
        self._gauges = {}
        self._histograms = {}
        self._last_values = {}

        if self.enabled:
            self.resource = Resource.create(
                {
                    "service.name": service_name,
                    "service.version": service_version,
                    "deployment.environment": environment,
                }
            )
            self._setup_opentelemetry()

    def _setup_opentelemetry(self):
        trace_exporter, metric_exporter = self._configure_exporters()
        self._setup_tracing(trace_exporter)
        self._setup_metrics(metric_exporter)
        RequestsInstrumentor().instrument()

    def _configure_exporters(self):
        return OTLPSpanExporter(), OTLPMetricExporter()

    def _setup_tracing(self, trace_exporter):
        tracer_provider = TracerProvider(resource=self.resource)
        trace.set_tracer_provider(tracer_provider)
        tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
        self.tracer = trace.get_tracer(self.service_name)

    def _setup_metrics(self, metric_exporter):
        reader = PeriodicExportingMetricReader(
            metric_exporter, export_interval_millis=10000
        )
        metrics_provider = MeterProvider(
            resource=self.resource, metric_readers=[reader]
        )
        metrics.set_meter_provider(metrics_provider)
        self.meter = metrics.get_meter(self.service_name)

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        if not self.enabled or not self.tracer:
            yield None
            return
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, v)
            yield span

    def record_counter(
        self, name: str, value: int = 1, attributes: Optional[Dict[str, str]] = None
    ):
        if not self.enabled or not self.meter:
            return
        if name not in self._counters:
            self._counters[name] = self.meter.create_counter(
                name, description=f"Counter for {name}", unit="1"
            )
        self._counters[name].add(value, attributes or {})

    def record_gauge(
        self, name: str, value: float, attributes: Optional[Dict[str, str]] = None
    ):
        if not self.enabled or not self.meter:
            return
        key = (name, str(attributes))
        if name not in self._gauges:
            self._gauges[name] = self.meter.create_up_down_counter(
                name, description=f"Gauge for {name}", unit="1"
            )
        last = self._last_values.get(key, 0)
        if last:
            self._gauges[name].add(-last, attributes or {})
        self._gauges[name].add(value, attributes or {})
        self._last_values[key] = value

    def record_histogram(
        self, name: str, value: float, attributes: Optional[Dict[str, str]] = None
    ):
        if not self.enabled or not self.meter:
            return
        if name not in self._histograms:
            self._histograms[name] = self.meter.create_histogram(
                name, description=f"Histogram for {name}", unit="ms"
            )
        self._histograms[name].record(value, attributes or {})
