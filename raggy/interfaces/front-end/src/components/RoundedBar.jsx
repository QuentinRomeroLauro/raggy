import { animated } from '@react-spring/web';
import { createElement, useCallback, useMemo } from 'react';
import { useTheme } from '@nivo/core';
import { useTooltip } from '@nivo/tooltip';
import { to } from '@react-spring/web';

// Custom bar modified to display border radius only at the top
export const RoundedBar = ({
    bar: { data, ...bar },
  
    style: {
      borderColor,
      color,
      height,
      labelColor,
      labelOpacity,
      labelX,
      labelY,
      transform,
      width
    },
  
    borderRadius,
    borderWidth,
  
    label,
    shouldRenderLabel,
  
    isInteractive,
    onClick,
    onMouseEnter,
    onMouseLeave,
  
    tooltip,
  
    isFocusable,
    ariaLabel,
    ariaLabelledBy,
    ariaDescribedBy
  }) => {
    const theme = useTheme();
    const { showTooltipFromEvent, showTooltipAt, hideTooltip } = useTooltip();
  
    const renderTooltip = useMemo(
      () => () => createElement(tooltip, { ...bar, ...data }),
      [tooltip, bar, data]
    );
  
    const handleClick = useCallback(
      (event) => {
        onClick?.({ color: bar.color, ...data }, event);
      },
      [bar, data, onClick]
    );
    const handleTooltip = useCallback(
      (event) =>
        showTooltipFromEvent(renderTooltip(), event),
      [showTooltipFromEvent, renderTooltip]
    );
    const handleMouseEnter = useCallback(
      (event) => {
        onMouseEnter?.(data, event);
        showTooltipFromEvent(renderTooltip(), event);
      },
      [data, onMouseEnter, showTooltipFromEvent, renderTooltip]
    );
    const handleMouseLeave = useCallback(
      (event) => {
        onMouseLeave?.(data, event);
        hideTooltip();
      },
      [data, hideTooltip, onMouseLeave]
    );
  
    // extra handlers to allow keyboard navigation
    const handleFocus = useCallback(() => {
      showTooltipAt(renderTooltip(), [bar.absX + bar.width / 2, bar.absY]);
    }, [showTooltipAt, renderTooltip, bar]);
    const handleBlur = useCallback(() => {
      hideTooltip();
    }, [hideTooltip]);
  
    return (
      <animated.g transform={transform}>
        <defs>
          <clipPath id={`round-corner-${label}`}>
            <animated.rect
              x="0"
              y="0"
              rx={borderRadius}
              ry={borderRadius}
              width={to(width, (value) => Math.max(value, 0))}
              height={to(height, (value) => Math.max(value + borderRadius, 0))}
            />
          </clipPath>
        </defs>
  
        <animated.rect
          clipPath={`url(#round-corner-${label})`}
          width={to(width, (value) => Math.max(value, 0))}
          height={to(height, (value) => Math.max(value, 0))}
          // rx={borderRadius}
          // ry={borderRadius}
          fill={data.fill ?? color}
          strokeWidth={borderWidth}
          stroke={borderColor}
          focusable={isFocusable}
          tabIndex={isFocusable ? 0 : undefined}
          aria-label={ariaLabel ? ariaLabel(data) : undefined}
          aria-labelledby={ariaLabelledBy ? ariaLabelledBy(data) : undefined}
          aria-describedby={ariaDescribedBy ? ariaDescribedBy(data) : undefined}
          onMouseEnter={isInteractive ? handleMouseEnter : undefined}
          onMouseMove={isInteractive ? handleTooltip : undefined}
          onMouseLeave={isInteractive ? handleMouseLeave : undefined}
          onClick={isInteractive ? handleClick : undefined}
          onFocus={isInteractive && isFocusable ? handleFocus : undefined}
          onBlur={isInteractive && isFocusable ? handleBlur : undefined}
        />
  
        {shouldRenderLabel && (
          <animated.text
            x={labelX}
            y={labelY}
            textAnchor="middle"
            dominantBaseline="central"
            fillOpacity={labelOpacity}
            style={{
              ...theme.labels.text,
              pointerEvents: 'none',
              fill: labelColor
            }}
          >
            {label}
          </animated.text>
        )}
      </animated.g>
    );
  };
