//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2023.05.17 at 07:25:31 PM CST 
//


package com.microsoft.Malmo.Schemas;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
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
 *       &lt;all>
 *         &lt;element name="ModifierList" minOccurs="0">
 *           &lt;complexType>
 *             &lt;complexContent>
 *               &lt;restriction base="{http://ProjectMalmo.microsoft.com}CommandListModifier">
 *                 &lt;choice maxOccurs="unbounded">
 *                   &lt;element name="command" type="{http://ProjectMalmo.microsoft.com}SimpleCraftCommand" maxOccurs="unbounded" minOccurs="0"/>
 *                 &lt;/choice>
 *               &lt;/restriction>
 *             &lt;/complexContent>
 *           &lt;/complexType>
 *         &lt;/element>
 *       &lt;/all>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {

})
@XmlRootElement(name = "SimpleCraftCommands")
public class SimpleCraftCommands {

    @XmlElement(name = "ModifierList")
    protected SimpleCraftCommands.ModifierList modifierList;

    /**
     * Gets the value of the modifierList property.
     * 
     * @return
     *     possible object is
     *     {@link SimpleCraftCommands.ModifierList }
     *     
     */
    public SimpleCraftCommands.ModifierList getModifierList() {
        return modifierList;
    }

    /**
     * Sets the value of the modifierList property.
     * 
     * @param value
     *     allowed object is
     *     {@link SimpleCraftCommands.ModifierList }
     *     
     */
    public void setModifierList(SimpleCraftCommands.ModifierList value) {
        this.modifierList = value;
    }


    /**
     * <p>Java class for anonymous complex type.
     * 
     * <p>The following schema fragment specifies the expected content contained within this class.
     * 
     * <pre>
     * &lt;complexType>
     *   &lt;complexContent>
     *     &lt;restriction base="{http://ProjectMalmo.microsoft.com}CommandListModifier">
     *       &lt;choice maxOccurs="unbounded">
     *         &lt;element name="command" type="{http://ProjectMalmo.microsoft.com}SimpleCraftCommand" maxOccurs="unbounded" minOccurs="0"/>
     *       &lt;/choice>
     *     &lt;/restriction>
     *   &lt;/complexContent>
     * &lt;/complexType>
     * </pre>
     * 
     * 
     */
    @XmlAccessorType(XmlAccessType.FIELD)
    @XmlType(name = "")
    public static class ModifierList
        extends CommandListModifier
    {


    }

}
