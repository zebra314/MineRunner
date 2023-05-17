//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2023.05.17 at 07:25:31 PM CST 
//


package com.microsoft.Malmo.Schemas;

import java.math.BigDecimal;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for anonymous complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType>
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="Width" type="{http://www.w3.org/2001/XMLSchema}int"/>
 *         &lt;element name="Height" type="{http://www.w3.org/2001/XMLSchema}int"/>
 *         &lt;element name="DepthScaling" minOccurs="0">
 *           &lt;complexType>
 *             &lt;complexContent>
 *               &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *                 &lt;attribute name="min" default="0">
 *                   &lt;simpleType>
 *                     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}decimal">
 *                       &lt;minInclusive value="0"/>
 *                       &lt;maxInclusive value="1"/>
 *                     &lt;/restriction>
 *                   &lt;/simpleType>
 *                 &lt;/attribute>
 *                 &lt;attribute name="max" default="1">
 *                   &lt;simpleType>
 *                     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}decimal">
 *                       &lt;minInclusive value="0"/>
 *                       &lt;maxInclusive value="1"/>
 *                     &lt;/restriction>
 *                   &lt;/simpleType>
 *                 &lt;/attribute>
 *                 &lt;attribute name="autoscale" type="{http://www.w3.org/2001/XMLSchema}boolean" default="true" />
 *               &lt;/restriction>
 *             &lt;/complexContent>
 *           &lt;/complexType>
 *         &lt;/element>
 *       &lt;/sequence>
 *       &lt;attribute name="want_depth" type="{http://www.w3.org/2001/XMLSchema}boolean" default="false" />
 *       &lt;attribute name="viewpoint" default="0">
 *         &lt;simpleType>
 *           &lt;restriction base="{http://www.w3.org/2001/XMLSchema}int">
 *             &lt;minInclusive value="0"/>
 *             &lt;maxInclusive value="2"/>
 *           &lt;/restriction>
 *         &lt;/simpleType>
 *       &lt;/attribute>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
    "width",
    "height",
    "depthScaling"
})
@XmlRootElement(name = "VideoProducer")
public class VideoProducer {

    @XmlElement(name = "Width")
    protected int width;
    @XmlElement(name = "Height")
    protected int height;
    @XmlElement(name = "DepthScaling")
    protected VideoProducer.DepthScaling depthScaling;
    @XmlAttribute(name = "want_depth")
    protected Boolean wantDepth;
    @XmlAttribute(name = "viewpoint")
    protected Integer viewpoint;

    /**
     * Gets the value of the width property.
     * 
     */
    public int getWidth() {
        return width;
    }

    /**
     * Sets the value of the width property.
     * 
     */
    public void setWidth(int value) {
        this.width = value;
    }

    /**
     * Gets the value of the height property.
     * 
     */
    public int getHeight() {
        return height;
    }

    /**
     * Sets the value of the height property.
     * 
     */
    public void setHeight(int value) {
        this.height = value;
    }

    /**
     * Gets the value of the depthScaling property.
     * 
     * @return
     *     possible object is
     *     {@link VideoProducer.DepthScaling }
     *     
     */
    public VideoProducer.DepthScaling getDepthScaling() {
        return depthScaling;
    }

    /**
     * Sets the value of the depthScaling property.
     * 
     * @param value
     *     allowed object is
     *     {@link VideoProducer.DepthScaling }
     *     
     */
    public void setDepthScaling(VideoProducer.DepthScaling value) {
        this.depthScaling = value;
    }

    /**
     * Gets the value of the wantDepth property.
     * 
     * @return
     *     possible object is
     *     {@link Boolean }
     *     
     */
    public boolean isWantDepth() {
        if (wantDepth == null) {
            return false;
        } else {
            return wantDepth;
        }
    }

    /**
     * Sets the value of the wantDepth property.
     * 
     * @param value
     *     allowed object is
     *     {@link Boolean }
     *     
     */
    public void setWantDepth(Boolean value) {
        this.wantDepth = value;
    }

    /**
     * Gets the value of the viewpoint property.
     * 
     * @return
     *     possible object is
     *     {@link Integer }
     *     
     */
    public int getViewpoint() {
        if (viewpoint == null) {
            return  0;
        } else {
            return viewpoint;
        }
    }

    /**
     * Sets the value of the viewpoint property.
     * 
     * @param value
     *     allowed object is
     *     {@link Integer }
     *     
     */
    public void setViewpoint(Integer value) {
        this.viewpoint = value;
    }


    /**
     * <p>Java class for anonymous complex type.
     * 
     * <p>The following schema fragment specifies the expected content contained within this class.
     * 
     * <pre>
     * &lt;complexType>
     *   &lt;complexContent>
     *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
     *       &lt;attribute name="min" default="0">
     *         &lt;simpleType>
     *           &lt;restriction base="{http://www.w3.org/2001/XMLSchema}decimal">
     *             &lt;minInclusive value="0"/>
     *             &lt;maxInclusive value="1"/>
     *           &lt;/restriction>
     *         &lt;/simpleType>
     *       &lt;/attribute>
     *       &lt;attribute name="max" default="1">
     *         &lt;simpleType>
     *           &lt;restriction base="{http://www.w3.org/2001/XMLSchema}decimal">
     *             &lt;minInclusive value="0"/>
     *             &lt;maxInclusive value="1"/>
     *           &lt;/restriction>
     *         &lt;/simpleType>
     *       &lt;/attribute>
     *       &lt;attribute name="autoscale" type="{http://www.w3.org/2001/XMLSchema}boolean" default="true" />
     *     &lt;/restriction>
     *   &lt;/complexContent>
     * &lt;/complexType>
     * </pre>
     * 
     * 
     */
    @XmlAccessorType(XmlAccessType.FIELD)
    @XmlType(name = "")
    public static class DepthScaling {

        @XmlAttribute(name = "min")
        protected BigDecimal min;
        @XmlAttribute(name = "max")
        protected BigDecimal max;
        @XmlAttribute(name = "autoscale")
        protected Boolean autoscale;

        /**
         * Gets the value of the min property.
         * 
         * @return
         *     possible object is
         *     {@link BigDecimal }
         *     
         */
        public BigDecimal getMin() {
            if (min == null) {
                return new BigDecimal("0");
            } else {
                return min;
            }
        }

        /**
         * Sets the value of the min property.
         * 
         * @param value
         *     allowed object is
         *     {@link BigDecimal }
         *     
         */
        public void setMin(BigDecimal value) {
            this.min = value;
        }

        /**
         * Gets the value of the max property.
         * 
         * @return
         *     possible object is
         *     {@link BigDecimal }
         *     
         */
        public BigDecimal getMax() {
            if (max == null) {
                return new BigDecimal("1");
            } else {
                return max;
            }
        }

        /**
         * Sets the value of the max property.
         * 
         * @param value
         *     allowed object is
         *     {@link BigDecimal }
         *     
         */
        public void setMax(BigDecimal value) {
            this.max = value;
        }

        /**
         * Gets the value of the autoscale property.
         * 
         * @return
         *     possible object is
         *     {@link Boolean }
         *     
         */
        public boolean isAutoscale() {
            if (autoscale == null) {
                return true;
            } else {
                return autoscale;
            }
        }

        /**
         * Sets the value of the autoscale property.
         * 
         * @param value
         *     allowed object is
         *     {@link Boolean }
         *     
         */
        public void setAutoscale(Boolean value) {
            this.autoscale = value;
        }

    }

}
